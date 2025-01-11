#include "schedulers/rl/rl_scheduler.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include "rl_scheduler.h"

namespace ghost {

pid_t pidOf(pid_t tid) {
  std::string statusFilePath = "/proc/" + std::to_string(tid) + "/status";
  std::ifstream statusFile(statusFilePath);
  std::string line;
  pid_t pid;
  while (std::getline(statusFile, line)) {
    if (line.substr(0, 4) == "Tgid") {
      std::istringstream iss(line);
      std::string key;
      iss >> key >> pid;
      break;
    }
  }
  statusFile.close();
  return pid;
}

RlScheduler::RlScheduler(Enclave* enclave, CpuList cpulist,
                             std::shared_ptr<TaskAllocator<RlTask>> allocator)
    : BasicDispatchScheduler(enclave, std::move(cpulist),
                             std::move(allocator)) {
  for (const Cpu& cpu : cpus()) {
    // TODO: extend Cpu to get numa node.
    int node = 0;
    CpuState* cs = cpu_state(cpu);
    cs->channel = enclave->MakeChannel(GHOST_MAX_QUEUE_ELEMS, node,
                                       MachineTopology()->ToCpuList({cpu}));
    // This channel pointer is valid for the lifetime of RlScheduler
    if (!default_channel_) {
      default_channel_ = cs->channel.get();
    }
  }

  CHECK_EQ(InitServerSocket(), 0);
}

RlScheduler::~RlScheduler() {
  if (server_socket_ != -1) {
    close(server_socket_);
    server_socket_ = -1;
  }
}

void RlScheduler::DumpAllTasks() {
  fprintf(stderr, "task        state   cpu\n");
  allocator()->ForEachTask([](Gtid gtid, const RlTask* task) {
    absl::FPrintF(stderr, "%-12s%-8d%-8d%c |%lu|%lu|%lu|%lu\n", gtid.describe(),
                  task->run_state, task->cpu, task->preempted ? 'P' : '-',
                  task->utime, task->stime, task->guest_time, task->vsize);
    return true;
  });
}

void RlScheduler::DumpState(const Cpu& cpu, int flags) {
  if (flags & Scheduler::kDumpAllTasks) {
    DumpAllTasks();
  }

  CpuState* cs = cpu_state(cpu);
  if (!(flags & Scheduler::kDumpStateEmptyRQ) && !cs->current &&
      cs->run_queue.Empty()) {
    return;
  }

  const RlTask* current = cs->current;
  const RlRq* rq = &cs->run_queue;
  absl::FPrintF(stderr, "SchedState[%d]: %s rq_l=%lu\n", cpu.id(),
                current ? current->gtid.describe() : "none", rq->Size());
}

void RlScheduler::EnclaveReady() {
  for (const Cpu& cpu : cpus()) {
    CpuState* cs = cpu_state(cpu);
    Agent* agent = enclave()->GetAgent(cpu);

    // AssociateTask may fail if agent barrier is stale.
    while (!cs->channel->AssociateTask(agent->gtid(), agent->barrier(),
                                       /*status=*/nullptr)) {
      CHECK_EQ(errno, ESTALE);
    }
  }
}

// Implicitly thread-safe because it is only called from one agent associated
// with the default queue.
Cpu RlScheduler::AssignCpu(RlTask* task) {
  static auto begin = cpus().begin();
  static auto end = cpus().end();
  static auto next = end;

  if (next == end) {
    next = begin;
  }
  return next++;
}

void RlScheduler::Migrate(RlTask* task, Cpu cpu, BarrierToken seqnum) {
  CHECK_EQ(task->run_state, RlTaskState::kRunnable);
  CHECK_EQ(task->cpu, -1);

  CpuState* cs = cpu_state(cpu);
  const Channel* channel = cs->channel.get();
  CHECK(channel->AssociateTask(task->gtid, seqnum, /*status=*/nullptr));

  GHOST_DPRINT(3, stderr, "Migrating task %s to cpu %d", task->gtid.describe(),
               cpu.id());
  task->cpu = cpu.id();

  // Make task visible in the new runqueue *after* changing the association
  // (otherwise the task can get oncpu while producing into the old queue).
  cs->run_queue.Enqueue(task);

  // Get the agent's attention so it notices the new task.
  enclave()->GetAgent(cpu)->Ping();
}

void RlScheduler::MigrateWithHint(RlTask* task, Cpu cpu, BarrierToken seqnum, SentCallbackType callback_type) {
  CHECK_EQ(task->run_state, RlTaskState::kRunnable);
  CHECK_EQ(task->cpu, -1);

  CpuState* cs = cpu_state(cpu);
  const Channel* channel = cs->channel.get();
  CHECK(channel->AssociateTask(task->gtid, seqnum, /*status=*/nullptr));

  GHOST_DPRINT(3, stderr, "Migrating task %s to cpu %d", task->gtid.describe(),
               cpu.id());
  task->cpu = cpu.id();

  cs->run_queue.EnqueueWithHint(task, this, callback_type);

  // Get the agent's attention so it notices the new task.
  enclave()->GetAgent(cpu)->Ping();
}

void RlScheduler::UpdateTask(RlTask* task, std::ifstream& instream) {
  const int utime_offset = 13;
  const int stime_offset = 14;
  const int guest_time_offset = 42;
  const int vsize_offset = 22;
  const int max_offset = std::max(
    std::max(utime_offset, stime_offset),
    std::max(guest_time_offset, vsize_offset)
  );
  std::string token;
  for(int i = 0; i <= max_offset; ++i) {
    instream >> token;
    switch (i) {
    case utime_offset:
      sscanf(token.c_str(), "%lu", &(task->utime));
      break;
    case stime_offset:
      sscanf(token.c_str(), "%lu", &(task->stime));
      break;
    case guest_time_offset:
      sscanf(token.c_str(), "%lu", &(task->guest_time));
      break;
    case vsize_offset:
      sscanf(token.c_str(), "%lu", &(task->vsize));
      break;
    default:
      break;
    }
  }
}

int RlScheduler::UpdateTaskFromStatFile(RlTask* task, pid_t tid) {
  pid_t pid = pidOf(tid);
  std::string stat_file_path = "/proc/" + std::to_string(pid) + "/task/" + std::to_string(tid) + "/stat";
  GHOST_DPRINT(5, stderr, "Reading file %s", stat_file_path);
  std::ifstream statFile(stat_file_path);
  if (!statFile.is_open()) {
    GHOST_DPRINT(3, stderr, "Failed to open /proc/%d/task/%d/stat", pid, tid);
    return 1;
  } else {
    this->UpdateTask(task, statFile);
    statFile.close();
    return 0;
  }
}

#if __BIG_ENDIAN__
# define htonll(x) (x)
#else
# define htonll(x) (((uint64_t)htonl((x) & 0xFFFFFFFF) << 32) | htonl((x) >> 32))
#endif

int RlScheduler::InitServerSocket() {
  server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
  if (server_socket_ == -1) {
    GHOST_DPRINT(1, stderr, "Failed to create server socket.");
    return 1;
  }

  int reuseaddr_value = 1;  
  if (setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &reuseaddr_value, sizeof(reuseaddr_value)) < 0) {
    GHOST_DPRINT(1, stderr, "Failed to set SO_REUSEADDR on server socket.");
    close(server_socket_);
    server_socket_ = -1;
    return 1;
  }

  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 20000;
  if (setsockopt(server_socket_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0) {
    GHOST_DPRINT(1, stderr, "Failed to set socket send timeout.");
    close(server_socket_);
    server_socket_ = -1;
    return 1;
  }
  if (setsockopt(server_socket_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
    GHOST_DPRINT(1, stderr, "Failed to set socket receive timeout.");
    close(server_socket_);
    server_socket_ = -1;
    return 1;
  }

  sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(listen_socket_port_);
  server_addr.sin_addr.s_addr = INADDR_ANY;

  if (bind(server_socket_, (sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
    GHOST_DPRINT(1, stderr, "Failed to bind server socket.");
    close(server_socket_);
    server_socket_ = -1;
    return 1;
  }

  if (listen(server_socket_, 1) == -1) {
    GHOST_DPRINT(1, stderr, "Failed to listen on server socket.");
    close(server_socket_);
    server_socket_ = -1;
    return 1;
  }

  GHOST_DPRINT(3, stderr, "Server socket initialized and listening on port %u.", listen_socket_port_);
  return 0;
}

int HandleClient(int client_socket, uint32_t* buf) {
  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 20000;
  if (setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
    GHOST_DPRINT(3, stderr, "Failed to set socket receive timeout. HandleClient aborted.");
    close(client_socket);
    return 1;
  }
  uint32_t num;
  ssize_t received_bytes = recv(client_socket, &num, sizeof(num), 0);
  if (received_bytes != sizeof(num)) {
    GHOST_DPRINT(3, stderr, "Failed to receive data. HandleClient aborted.");
    close(client_socket);
    return 1;
  }
  *buf = ntohl(num);
  close(client_socket);
  return 0;
}

int RlScheduler::ReceiveAction(uint32_t* buf) {
  CHECK_NE(server_socket_, -1);

  GHOST_DPRINT(3, stderr, "ReceiveAction: waiting for environment.");
  int client_socket = accept(server_socket_, nullptr, nullptr);
  if (client_socket == -1) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      GHOST_DPRINT(3, stderr, "Socket connection timed out. ReceiveAction aborted.");
    } else {
      GHOST_DPRINT(3, stderr, "Failed to accept connection: %s. ReceiveAction aborted.", strerror(errno));
    }
    return 1;
  }

  int status = HandleClient(client_socket, buf);
  return status;
}

int SendSequence(std::vector<uint64_t>& sequence, uint16_t port) {
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock == -1) {
    GHOST_DPRINT(3, stderr, "Failed to create socket. SendSequence aborted.");
    return 1;
  }

  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 20000;
  if (setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0) {
    GHOST_DPRINT(3, stderr, "Failed to set socket send timeout. SendSequence aborted.");
    return 1;
  }
  if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
    GHOST_DPRINT(3, stderr, "Failed to set socket receive timeout. SendSequence aborted.");
    return 1;
  }

  sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port);
  inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);
  if (connect(sock, (sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
    GHOST_DPRINT(3, stderr, "Failed to connect. SendSequence aborted.");
    close(sock);
    return 1;
  }

  uint32_t length = htonl(sequence.size());
  if (send(sock, &length, sizeof(length), 0) == -1) {
    GHOST_DPRINT(3, stderr, "Failed to send. SendSequence aborted.");
    close(sock);
    return 1;
  }    
  for (int i = 0; i < sequence.size(); ++i) {
    uint64_t net_num = htonll(sequence[i]);
    sequence[i] = net_num;
  }
  if (send(sock, sequence.data(), sequence.size() * sizeof(uint64_t), 0) == -1) {
    GHOST_DPRINT(3, stderr, "Failed to send. SendSequence aborted.");
    close(sock);
    return 1;
  }
  close(sock);
  return 0;
}

void AddTaskInfoToMetrics(const RlTask* task, std::vector<uint64_t>& vec) {
  vec.push_back((uint64_t)task->run_state);
  vec.push_back((uint64_t)task->cpu);
  vec.push_back((uint64_t)task->preempted);
  vec.push_back((uint64_t)task->utime);
  vec.push_back((uint64_t)task->stime);
  vec.push_back((uint64_t)task->guest_time);
  vec.push_back((uint64_t)task->vsize);
}

int RlScheduler::ShareTask(const RlTask* task, const SentCallbackType callback_type, bool action_expected) {
  std::vector<uint64_t> metrics = std::vector<uint64_t>();
  if (action_expected) {
    metrics.push_back((uint64_t)1);
  } else {
    metrics.push_back((uint64_t)0);
  }
  metrics.push_back((uint64_t)callback_type);
  AddTaskInfoToMetrics(task, metrics);
  if (task->cpu >= 0) {
    const std::deque<RlTask*> run_queue_content = cpu_state_of(task)->run_queue.dump();
    for (RlTask* cur_task : run_queue_content) {
      AddTaskInfoToMetrics(cur_task, metrics);
    }
  }
  return SendSequence(metrics, this->target_socket_port_);
}

void RlScheduler::TaskNew(RlTask* task, const Message& msg) {
  const ghost_msg_payload_task_new* payload =
      static_cast<const ghost_msg_payload_task_new*>(msg.payload());

  task->seqnum = msg.seqnum();
  task->run_state = RlTaskState::kBlocked;

  if (task->NeedsInfoUpdate(msg)) {
    pid_t tid = Gtid(payload->gtid).tid();
    if(this->UpdateTaskFromStatFile(task, tid) == 0) {
      GHOST_DPRINT(4, stderr, "Updated task %s info successfully", task->gtid.describe());
    }
  }

  if (payload->runnable) {
    task->run_state = RlTaskState::kRunnable;
    Cpu cpu = AssignCpu(task);
    MigrateWithHint(task, cpu, msg.seqnum(), SentCallbackType::kTaskNew);
  } else {
    // Wait until task becomes runnable to avoid race between migration
    // and MSG_TASK_WAKEUP showing up on the default channel.
  }
}

void RlScheduler::TaskRunnable(RlTask* task, const Message& msg) {
  const ghost_msg_payload_task_wakeup* payload =
      static_cast<const ghost_msg_payload_task_wakeup*>(msg.payload());

  CHECK(task->blocked());
  task->run_state = RlTaskState::kRunnable;

  // A non-deferrable wakeup gets the same preference as a preempted task.
  // This is because it may be holding locks or resources needed by other
  // tasks to make progress.
  task->prio_boost = !payload->deferrable;

  if (task->NeedsInfoUpdate(msg)) {
    pid_t tid = Gtid(payload->gtid).tid();
    if(this->UpdateTaskFromStatFile(task, tid) == 0) {
      GHOST_DPRINT(4, stderr, "Updated task %s info successfully", task->gtid.describe());
    }
  }

  if (task->cpu < 0) {
    // There cannot be any more messages pending for this task after a
    // MSG_TASK_WAKEUP (until the agent puts it oncpu) so it's safe to
    // migrate.
    Cpu cpu = AssignCpu(task);
    MigrateWithHint(task, cpu, msg.seqnum(), SentCallbackType::kTaskRunnable);
  } else {
    CpuState* cs = cpu_state_of(task);
    if (this->ShareTask(task, SentCallbackType::kTaskRunnable, false)) {
      GHOST_DPRINT(2, stderr, 
                  "Failed to send information about non-actionable callback for %s.",
                  task->gtid.describe());
    }
    cs->run_queue.Enqueue(task);
  }
}

void RlScheduler::TaskDeparted(RlTask* task, const Message& msg) {
  const ghost_msg_payload_task_departed* payload =
      static_cast<const ghost_msg_payload_task_departed*>(msg.payload());

  if (task->oncpu() || payload->from_switchto) {
    TaskOffCpu(task, /*blocked=*/false, payload->from_switchto);
  } else if (task->queued()) {
    CpuState* cs = cpu_state_of(task);
    cs->run_queue.Erase(task);
  } else {
    CHECK(task->blocked());
  }

  if (payload->from_switchto) {
    Cpu cpu = topology()->cpu(payload->cpu);
    enclave()->GetAgent(cpu)->Ping();
  }

  // Not updating task data from statfile when task departs.

  if (this->ShareTask(task, SentCallbackType::kTaskDeparted, false)) {
    GHOST_DPRINT(2, stderr, 
                  "Failed to send information about non-actionable callback for %s.",
                  task->gtid.describe());
  }
  allocator()->FreeTask(task);
}

void RlScheduler::TaskDead(RlTask* task, const Message& msg) {
  CHECK(task->blocked());

  // Not updating task data from statfile when task dies.

  if (this->ShareTask(task, SentCallbackType::kTaskDead, false)) {
    GHOST_DPRINT(2, stderr, 
                  "Failed to send information about non-actionable callback for %s.",
                  task->gtid.describe());
  }
  allocator()->FreeTask(task);
}

void RlScheduler::TaskYield(RlTask* task, const Message& msg) {
  const ghost_msg_payload_task_yield* payload =
      static_cast<const ghost_msg_payload_task_yield*>(msg.payload());

  TaskOffCpu(task, /*blocked=*/false, payload->from_switchto);

  if (task->NeedsInfoUpdate(msg)) {
    pid_t tid = Gtid(payload->gtid).tid();
    if(this->UpdateTaskFromStatFile(task, tid) == 0) {
      GHOST_DPRINT(4, stderr, "Updated task %s info successfully", task->gtid.describe());
    }
  }

  CpuState* cs = cpu_state_of(task);
  cs->run_queue.EnqueueWithHint(task, this, SentCallbackType::kTaskYield);

  if (payload->from_switchto) {
    Cpu cpu = topology()->cpu(payload->cpu);
    enclave()->GetAgent(cpu)->Ping();
  }
}

void RlScheduler::TaskBlocked(RlTask* task, const Message& msg) {
  const ghost_msg_payload_task_blocked* payload =
      static_cast<const ghost_msg_payload_task_blocked*>(msg.payload());

  TaskOffCpu(task, /*blocked=*/true, payload->from_switchto);

  if (payload->from_switchto) {
    Cpu cpu = topology()->cpu(payload->cpu);
    enclave()->GetAgent(cpu)->Ping();
  }
}

void RlScheduler::TaskPreempted(RlTask* task, const Message& msg) {
  const ghost_msg_payload_task_preempt* payload =
      static_cast<const ghost_msg_payload_task_preempt*>(msg.payload());

  TaskOffCpu(task, /*blocked=*/false, payload->from_switchto);

  task->preempted = true;
  task->prio_boost = true;
  CpuState* cs = cpu_state_of(task);
  cs->run_queue.Enqueue(task);

  if (payload->from_switchto) {
    Cpu cpu = topology()->cpu(payload->cpu);
    enclave()->GetAgent(cpu)->Ping();
  }
}

void RlScheduler::TaskSwitchto(RlTask* task, const Message& msg) {
  TaskOffCpu(task, /*blocked=*/true, /*from_switchto=*/false);
}


void RlScheduler::TaskOffCpu(RlTask* task, bool blocked,
                               bool from_switchto) {
  GHOST_DPRINT(3, stderr, "Task %s offcpu %d", task->gtid.describe(),
               task->cpu);
  CpuState* cs = cpu_state_of(task);

  if (task->oncpu()) {
    CHECK_EQ(cs->current, task);
    cs->current = nullptr;
  } else {
    CHECK(from_switchto);
    CHECK_EQ(task->run_state, RlTaskState::kBlocked);
  }

  task->run_state =
      blocked ? RlTaskState::kBlocked : RlTaskState::kRunnable;
}

void RlScheduler::TaskOnCpu(RlTask* task, Cpu cpu) {
  CpuState* cs = cpu_state(cpu);
  cs->current = task;

  GHOST_DPRINT(3, stderr, "Task %s oncpu %d", task->gtid.describe(), cpu.id());

  task->run_state = RlTaskState::kOnCpu;
  task->cpu = cpu.id();
  task->preempted = false;
  task->prio_boost = false;
}

void RlScheduler::RlSchedule(const Cpu& cpu, BarrierToken agent_barrier,
                                 bool prio_boost) {
  CpuState* cs = cpu_state(cpu);
  RlTask* next = nullptr;
  if (!prio_boost) {
    next = cs->current;
    if (!next) next = cs->run_queue.Dequeue();
  }

  GHOST_DPRINT(3, stderr, "RlSchedule %s on %s cpu %d ",
               next ? next->gtid.describe() : "idling",
               prio_boost ? "prio-boosted" : "", cpu.id());

  RunRequest* req = enclave()->GetRunRequest(cpu);
  if (next) {
    // Wait for 'next' to get offcpu before switching to it. This might seem
    // superfluous because we don't migrate tasks past the initial assignment
    // of the task to a cpu. However a SwitchTo target can migrate and run on
    // another CPU behind the agent's back. This is usually undetectable from
    // the agent's pov since the SwitchTo target is blocked and thus !on_rq.
    //
    // However if 'next' happens to be the last task in a SwitchTo chain then
    // it is possible to process TASK_WAKEUP(next) before it has gotten off
    // the remote cpu. The 'on_cpu()' check below handles this scenario.
    //
    // See go/switchto-ghost for more details.
    while (next->status_word.on_cpu()) {
      Pause();
    }

    req->Open({
        .target = next->gtid,
        .target_barrier = next->seqnum,
        .agent_barrier = agent_barrier,
        .commit_flags = COMMIT_AT_TXN_COMMIT,
    });

    if (req->Commit()) {
      // Txn commit succeeded and 'next' is oncpu.
      TaskOnCpu(next, cpu);
    } else {
      GHOST_DPRINT(3, stderr, "RlSchedule: commit failed (state=%d)",
                   req->state());

      if (next == cs->current) {
        TaskOffCpu(next, /*blocked=*/false, /*from_switchto=*/false);
      }

      // Txn commit failed so push 'next' to the front of runqueue.
      next->prio_boost = true;
      cs->run_queue.Enqueue(next);
    }
  } else {
    // If LocalYield is due to 'prio_boost' then instruct the kernel to
    // return control back to the agent when CPU is idle.
    int flags = 0;
    if (prio_boost && (cs->current || !cs->run_queue.Empty())) {
      flags = RTLA_ON_IDLE;
    }
    req->LocalYield(agent_barrier, flags);
  }
}

void RlScheduler::Schedule(const Cpu& cpu, const StatusWord& agent_sw) {
  BarrierToken agent_barrier = agent_sw.barrier();
  CpuState* cs = cpu_state(cpu);

  GHOST_DPRINT(3, stderr, "Schedule: agent_barrier[%d] = %d\n", cpu.id(),
               agent_barrier);

  Message msg;
  while (!(msg = Peek(cs->channel.get())).empty()) {
    DispatchMessage(msg);
    Consume(cs->channel.get(), msg);
  }

  RlSchedule(cpu, agent_barrier, agent_sw.boosted_priority());
}

void RlRq::Enqueue(RlTask* task) {
  CHECK_GE(task->cpu, 0);
  CHECK_EQ(task->run_state, RlTaskState::kRunnable);

  task->run_state = RlTaskState::kQueued;

  absl::MutexLock lock(&mu_);
  if (task->prio_boost)
    rq_.push_front(task);
  else
    rq_.push_back(task);
}

void RlRq::EnqueueTo(RlTask* task, uint32_t place, bool respect_prio_boost) {
  CHECK_GE(task->cpu, 0);
  CHECK_EQ(task->run_state, RlTaskState::kRunnable);

  task->run_state = RlTaskState::kQueued;

  absl::MutexLock lock(&mu_);
  if (respect_prio_boost && task->prio_boost) {
    rq_.push_front(task);
    GHOST_DPRINT(4, stderr, "Inserted: task at head is %s", rq_[0]->gtid.describe());
  }
  else {
    rq_.insert(rq_.begin() + place, task);
    GHOST_DPRINT(4, stderr, "Inserted: task at place %d is %s", place, rq_[place]->gtid.describe());
  }
}

void ghost::RlRq::EnqueueWithHint(RlTask *task, RlScheduler* communicator, 
                                  SentCallbackType callback_type, bool respect_prio_boost) {
  CHECK_GE(task->cpu, 0);
  CHECK_EQ(task->run_state, RlTaskState::kRunnable);

  task->run_state = RlTaskState::kQueued;

  absl::MutexLock lock(&mu_);
  if (communicator->ShareTask(task, callback_type, true)) {
    GHOST_DPRINT(2, stderr, 
                  "Failed to send information about actionable callback for %s. Doing default action.",
                  task->gtid.describe());
    if (task->prio_boost) {
      rq_.push_front(task);
    }
    else {
      rq_.push_back(task);
    }
    return;
  }
  uint32_t place;
  if (communicator->ReceiveAction(&place)) {
    GHOST_DPRINT(2, stderr, 
                  "Failed to receive action hint for %s in time. Doing default action.",
                  task->gtid.describe());
    if (task->prio_boost) {
      rq_.push_front(task);
    }
    else {
      rq_.push_back(task);
    }
    return;
  }

  GHOST_DPRINT(2, stderr, "Enqueue task %s to place %d in the queue", task->gtid.describe(), place);
  if (respect_prio_boost && task->prio_boost) {
    rq_.push_front(task);
    GHOST_DPRINT(4, stderr, "Inserted: task at head is %s", rq_[0]->gtid.describe());
  }
  else {
    rq_.insert(rq_.begin() + place, task);
    GHOST_DPRINT(4, stderr, "Inserted: task at place %d is %s", place, rq_[place]->gtid.describe());
  }
}

RlTask* RlRq::Dequeue() {
  absl::MutexLock lock(&mu_);
  if (rq_.empty()) return nullptr;

  RlTask* task = rq_.front();
  CHECK(task->queued()) << task->gtid.describe();
  task->run_state = RlTaskState::kRunnable;
  rq_.pop_front();
  return task;
}

void RlRq::Erase(RlTask* task) {
  CHECK_EQ(task->run_state, RlTaskState::kQueued);
  absl::MutexLock lock(&mu_);
  size_t size = rq_.size();
  if (size > 0) {
    // Check if 'task' is at the back of the runqueue (common case).
    size_t pos = size - 1;
    if (rq_[pos] == task) {
      rq_.erase(rq_.cbegin() + pos);
      task->run_state = RlTaskState::kRunnable;
      return;
    }

    // Now search for 'task' from the beginning of the runqueue.
    for (pos = 0; pos < size - 1; pos++) {
      if (rq_[pos] == task) {
        rq_.erase(rq_.cbegin() + pos);
        task->run_state =  RlTaskState::kRunnable;
        return;
      }
    }
  }
  CHECK(false);
}

std::unique_ptr<RlScheduler> MultiThreadedRlScheduler(Enclave* enclave,
                                                          CpuList cpulist) {
  auto allocator = std::make_shared<ThreadSafeMallocTaskAllocator<RlTask>>();
  auto scheduler = std::make_unique<RlScheduler>(enclave, std::move(cpulist),
                                                   std::move(allocator));
  return scheduler;
}

void RlAgent::AgentThread() {
  gtid().assign_name("Agent:" + std::to_string(cpu().id()));
  if (verbose() > 1) {
    printf("Agent tid:=%d\n", gtid().tid());
  }
  SignalReady();
  WaitForEnclaveReady();

  PeriodicEdge debug_out(absl::Seconds(1));

  while (!Finished() || !scheduler_->Empty(cpu())) {
    scheduler_->Schedule(cpu(), status_word());

    if (verbose() && debug_out.Edge()) {
      static const int flags = verbose() > 1 ? Scheduler::kDumpStateEmptyRQ : 0;
      if (scheduler_->debug_runqueue_) {
        scheduler_->debug_runqueue_ = false;
        scheduler_->DumpState(cpu(), Scheduler::kDumpAllTasks);
      } else {
        scheduler_->DumpState(cpu(), flags);
      }
    }
  }
}

std::ostream& operator<<(std::ostream& os, const RlTaskState& state) {
  switch (state) {
    case RlTaskState::kBlocked:
      return os << "kBlocked";
    case RlTaskState::kRunnable:
      return os << "kRunnable";
    case RlTaskState::kQueued:
      return os << "kQueued";
    case RlTaskState::kOnCpu:
      return os << "kOnCpu";
  }
}

}  //  namespace ghost

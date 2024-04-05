#include "schedulers/rl/rl_scheduler.h"
#include "shared/fd_server.h"

#include <memory>
#include <cstdio>
#include <cstdlib>
#include <sstream>
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

void RlScheduler::ShareTask(const RlTask* task) {
  std::stringstream command;
  command << "echo \"" << /* space-separated metrics */
    task->run_state << " " << task->cpu << " " << task->tid() << 
    " " << (task->preempted ? '1' : '0') << " " << task->utime << " " << task->stime << 
    " " << task->guest_time << " " << task->vsize << "\" | $FDSRV " << this->share_counter_++;
  if (char status = system(command.str().c_str())) {
    absl::FPrintF(stderr, "Share command failed with status code %d\n", status);
  }
}

void RlScheduler::TaskNew(RlTask* task, const Message& msg) {
  const ghost_msg_payload_task_new* payload =
      static_cast<const ghost_msg_payload_task_new*>(msg.payload());

  task->seqnum = msg.seqnum();
  task->run_state = RlTaskState::kBlocked;

  if (task->NeedsInfoUpdate(msg)) {
    pid_t tid = Gtid(payload->gtid).tid();
    pid_t pid = pidOf(tid);
    std::string stat_file_path = "/proc/" + std::to_string(pid) + "/task/" + std::to_string(tid) + "/stat";
    absl::FPrintF(stdout, "Reading file %s\n", stat_file_path);
    std::ifstream statFile(stat_file_path);
    if (!statFile.is_open()) {
      absl::FPrintF(stderr, "Failed to open /proc/%d/task/%d/stat\n", pid, tid);
    } else {
      this->UpdateTask(task, statFile);
      absl::FPrintF(stdout, "Updated task\n");
      statFile.close();
    }
  }

  if (payload->runnable) {
    task->run_state = RlTaskState::kRunnable;
    Cpu cpu = AssignCpu(task);
    Migrate(task, cpu, msg.seqnum());
  } else {
    // Wait until task becomes runnable to avoid race between migration
    // and MSG_TASK_WAKEUP showing up on the default channel.
  }
  this->ShareTask(task);
  // this->DumpAllTasks();
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

  if (task->cpu < 0) {
    // There cannot be any more messages pending for this task after a
    // MSG_TASK_WAKEUP (until the agent puts it oncpu) so it's safe to
    // migrate.
    Cpu cpu = AssignCpu(task);
    Migrate(task, cpu, msg.seqnum());
  } else {
    CpuState* cs = cpu_state_of(task);
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

  allocator()->FreeTask(task);
}

void RlScheduler::TaskDead(RlTask* task, const Message& msg) {
  CHECK(task->blocked());
  allocator()->FreeTask(task);
}

void RlScheduler::TaskYield(RlTask* task, const Message& msg) {
  const ghost_msg_payload_task_yield* payload =
      static_cast<const ghost_msg_payload_task_yield*>(msg.payload());

  TaskOffCpu(task, /*blocked=*/false, payload->from_switchto);

  CpuState* cs = cpu_state_of(task);
  cs->run_queue.Enqueue(task);

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

RlTask* RlRq::Dequeue() {
  absl::MutexLock lock(&mu_);
  if (rq_.empty()) return nullptr;

  RlTask* task = rq_.front();
  CHECK(task->queued());
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

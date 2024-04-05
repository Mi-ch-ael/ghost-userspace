#ifndef GHOST_SCHEDULERS_RL_RL_SCHEDULER_PRINT_H
#define GHOST_SCHEDULERS_RL_RL_SCHEDULER_PRINT_H

#include <deque>
#include <memory>

#include "lib/agent.h"
#include "lib/scheduler.h"

namespace ghost {

enum class RlTaskState {
  kBlocked,   // not on runqueue.
  kRunnable,  // transitory state:
              // 1. kBlocked->kRunnable->kQueued
              // 2. kQueued->kRunnable->kOnCpu
  kQueued,    // on runqueue.
  kOnCpu,     // running on cpu.
};

// For CHECK and friends.
std::ostream& operator<<(std::ostream& os, const RlTaskState& state);

pid_t pidOf(pid_t tid);

struct RlTask : public Task<> {
  explicit RlTask(Gtid rl_task_gtid, ghost_sw_info sw_info)
      : Task<>(rl_task_gtid, sw_info) {}
  ~RlTask() override {}

  inline bool blocked() const { return run_state == RlTaskState::kBlocked; }
  inline bool queued() const { return run_state == RlTaskState::kQueued; }
  inline bool oncpu() const { return run_state == RlTaskState::kOnCpu; }

  // Updating information read from `/proc` in every callback can turn out 
  // to be very taxing on performance. One might want to redefine this to
  // read updates from `/proc/pid` once in a while. For now, this always
  // returns `true`.
  bool NeedsInfoUpdate(const Message& msg) const { return true; } 

  pid_t tid() const { return this->gtid.tid(); }

  // N.B. _runnable() is a transitory state typically used during runqueue
  // manipulation. It is not expected to be used from task msg callbacks.
  //
  // If you are reading this then you probably want to take a closer look
  // at queued() instead.
  inline bool _runnable() const {
    return run_state == RlTaskState::kRunnable;
  }

  RlTaskState run_state = RlTaskState::kBlocked;
  int cpu = -1;

  // Whether the last execution was preempted or not.
  bool preempted = false;

  // https://man7.org/linux/man-pages/man5/proc.5.html

  u_long utime = 0;
  u_long stime = 0;
  u_long vsize = 0;
  u_long guest_time = 0;

  // A task's priority is boosted on a kernel preemption or a !deferrable
  // wakeup - basically when it may be holding locks or other resources
  // that prevent other tasks from making progress.
  // TODO: remove this when done with actual implementation.
  bool prio_boost = false;
};

class RlRq {
 public:
  RlRq() = default;
  RlRq(const RlRq&) = delete;
  RlRq& operator=(RlRq&) = delete;

  RlTask* Dequeue();
  void Enqueue(RlTask* task);

  // Erase 'task' from the runqueue.
  //
  // Caller must ensure that 'task' is on the runqueue in the first place
  // (e.g. via task->queued()).
  void Erase(RlTask* task);

  size_t Size() const {
    absl::MutexLock lock(&mu_);
    return rq_.size();
  }

  bool Empty() const { return Size() == 0; }

 private:
  mutable absl::Mutex mu_;
  std::deque<RlTask*> rq_ ABSL_GUARDED_BY(mu_);
};

class RlScheduler : public BasicDispatchScheduler<RlTask> {
 public:
  explicit RlScheduler(Enclave* enclave, CpuList cpulist,
                         std::shared_ptr<TaskAllocator<RlTask>> allocator);
  ~RlScheduler() final {}

  void Schedule(const Cpu& cpu, const StatusWord& sw);

  void EnclaveReady() final;
  Channel& GetDefaultChannel() final { return *default_channel_; };

  bool Empty(const Cpu& cpu) {
    CpuState* cs = cpu_state(cpu);
    return cs->run_queue.Empty();
  }

  void DumpState(const Cpu& cpu, int flags) final;
  std::atomic<bool> debug_runqueue_ = false;

  int CountAllTasks() {
    int num_tasks = 0;
    allocator()->ForEachTask([&num_tasks](Gtid gtid, const RlTask* task) {
      ++num_tasks;
      return true;
    });
    return num_tasks;
  }

  // Update `task` with stat data from `instream` 
  void UpdateTask(RlTask* task, std::ifstream& instream);
  // Share `task` via `FdServer` (essentially via a socket)
  // $FDSRV variable MUST be set
  // To set it, run `source setup.sh` in repository root directory
  // Note: agent needs root privileges to run, so run `source` as root
  void ShareTask(const RlTask* task);

  static constexpr int kDebugRunqueue = 1;
  static constexpr int kCountAllTasks = 2;

 protected:
  void TaskNew(RlTask* task, const Message& msg) final;
  void TaskRunnable(RlTask* task, const Message& msg) final;
  void TaskDeparted(RlTask* task, const Message& msg) final;
  void TaskDead(RlTask* task, const Message& msg) final;
  void TaskYield(RlTask* task, const Message& msg) final;
  void TaskBlocked(RlTask* task, const Message& msg) final;
  void TaskPreempted(RlTask* task, const Message& msg) final;
  void TaskSwitchto(RlTask* task, const Message& msg) final;

 private:
  void RlSchedule(const Cpu& cpu, BarrierToken agent_barrier,
                    bool prio_boosted);
  void TaskOffCpu(RlTask* task, bool blocked, bool from_switchto);
  void TaskOnCpu(RlTask* task, Cpu cpu);
  void Migrate(RlTask* task, Cpu cpu, BarrierToken seqnum);
  Cpu AssignCpu(RlTask* task);
  void DumpAllTasks();

  struct CpuState {
    RlTask* current = nullptr;
    std::unique_ptr<Channel> channel = nullptr;
    RlRq run_queue;
  } ABSL_CACHELINE_ALIGNED;

  inline CpuState* cpu_state(const Cpu& cpu) { return &cpu_states_[cpu.id()]; }

  inline CpuState* cpu_state_of(const RlTask* task) {
    CHECK_GE(task->cpu, 0);
    CHECK_LT(task->cpu, MAX_CPUS);
    return &cpu_states_[task->cpu];
  }

  CpuState cpu_states_[MAX_CPUS];
  Channel* default_channel_ = nullptr;

  int share_counter_ = 0;
};

std::unique_ptr<RlScheduler> MultiThreadedRlScheduler(Enclave* enclave,
                                                          CpuList cpulist);
class RlAgent : public LocalAgent {
 public:
  RlAgent(Enclave* enclave, Cpu cpu, RlScheduler* scheduler)
      : LocalAgent(enclave, cpu), scheduler_(scheduler) {}

  void AgentThread() override;
  Scheduler* AgentScheduler() const override { return scheduler_; }

 private:
  RlScheduler* scheduler_;
};

template <class EnclaveType>
class FullRlAgent : public FullAgent<EnclaveType> {
 public:
  explicit FullRlAgent(AgentConfig config) : FullAgent<EnclaveType>(config) {
    scheduler_ =
        MultiThreadedRlScheduler(&this->enclave_, *this->enclave_.cpus());
    this->StartAgentTasks();
    this->enclave_.Ready();
  }

  ~FullRlAgent() override {
    this->TerminateAgentTasks();
  }

  std::unique_ptr<Agent> MakeAgent(const Cpu& cpu) override {
    return std::make_unique<RlAgent>(&this->enclave_, cpu, scheduler_.get());
  }

  void RpcHandler(int64_t req, const AgentRpcArgs& args,
                  AgentRpcResponse& response) override {
    switch (req) {
      case RlScheduler::kDebugRunqueue:
        scheduler_->debug_runqueue_ = true;
        response.response_code = 0;
        return;
      case RlScheduler::kCountAllTasks:
        response.response_code = scheduler_->CountAllTasks();
        return;
      default:
        response.response_code = -1;
        return;
    }
  }

 private:
  std::unique_ptr<RlScheduler> scheduler_;
};

}  // namespace ghost

#endif  // GHOST_SCHEDULERS_RL_RL_SCHEDULER_PRINT_H

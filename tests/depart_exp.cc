#include <stdio.h>

#include <atomic>
#include <memory>
#include <vector>

#include "lib/base.h"
#include "lib/ghost.h"

namespace ghost {
namespace {

struct ScopedTime {
  ScopedTime() { start = absl::Now(); }
  ~ScopedTime() {
    printf(" took %0.2f ms\n", absl::ToDoubleMilliseconds(absl::Now() - start));
  }
  absl::Time start;
};

void TaskDeparted() {
  printf("\nStarting simple worker\n");
  GhostThread t(GhostThread::KernelScheduler::kGhost, [] {
    fprintf(stderr, "hello world!\n");
    absl::SleepFor(absl::Milliseconds(10));

    fprintf(stderr, "fantastic nap! departing ghOSt now for CFS...\n");
    const sched_param param{};
    CHECK_EQ(sched_setscheduler(/*pid=*/0, SCHED_OTHER, &param), 0);
    CHECK_EQ(sched_getscheduler(/*pid=*/0), SCHED_OTHER);
    fprintf(stderr, "hello from CFS!\n");
  });

  t.Join();
  printf("\nFinished simple worker\n");
}

void TaskDepartedMany(int num_threads) {
  std::vector<std::unique_ptr<GhostThread>> threads;

  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(
        new GhostThread(GhostThread::KernelScheduler::kGhost, [] {
          absl::SleepFor(absl::Milliseconds(10));

          const sched_param param{};
          CHECK_EQ(sched_setscheduler(/*pid=*/0, SCHED_OTHER, &param), 0);
          CHECK_EQ(sched_getscheduler(/*pid=*/0), SCHED_OTHER);
        }));
  }

  for (auto& t : threads) t->Join();
}

}  // namespace
}  // namespace ghost

int main() {
  {
    printf("TaskDeparted\n");
    ghost::ScopedTime time;
    ghost::TaskDeparted();
  }
  {
    printf("TaskDepartedMany\n");
    ghost::ScopedTime time;
    ghost::TaskDepartedMany(10);
  }
  return 0;
}

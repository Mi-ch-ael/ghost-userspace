#include <stdio.h>
#include <sched.h>

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

void YieldExp() {
  printf("\nStarting yield exp\n");
  GhostThread t(GhostThread::KernelScheduler::kGhost, [] {
    fprintf(stderr, "hello world!\n");
    sched_yield();
    fprintf(stderr, "back again!\n");
    // Verify that a ghost thread implicitly clones itself in the ghost
    // scheduling class.
    std::thread t2(
        [] { CHECK_EQ(sched_getscheduler(/*pid=*/0), SCHED_GHOST); });
    t2.join();
  });
  t.Join();
  printf("\nFinished yield exp\n");
}

void YieldExpMany(int num_threads) {
  std::vector<std::unique_ptr<GhostThread>> threads;

  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(
        new GhostThread(GhostThread::KernelScheduler::kGhost, [] {
          sched_yield();
          std::thread t2(
            [] { CHECK_EQ(sched_getscheduler(/*pid=*/0), SCHED_GHOST); });
          t2.join();
        }));
  }

  for (auto& t : threads) t->Join();
}

}  // namespace
}  // namespace ghost

int main() {
  {
    printf("YieldExp\n");
    ghost::ScopedTime time;
    ghost::YieldExp();
  }
  {
    printf("YieldExpMany\n");
    ghost::ScopedTime time;
    ghost::YieldExpMany(10);
  }
  return 0;
}

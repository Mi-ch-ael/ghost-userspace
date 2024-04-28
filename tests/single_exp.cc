#include <stdio.h>
#include <sched.h>

#include <atomic>
#include <memory>
#include <vector>

#include "lib/base.h"
#include "lib/ghost.h"

// A series of simple tests for ghOSt schedulers.

namespace ghost {
namespace {

struct ScopedTime {
  ScopedTime() { start = absl::Now(); }
  ~ScopedTime() {
    printf(" took %0.2f ms\n", absl::ToDoubleMilliseconds(absl::Now() - start));
  }
  absl::Time start;
};

void SimpleExp() {
  printf("\nStarting simple worker\n");
  GhostThread t(GhostThread::KernelScheduler::kGhost, [] {
    fprintf(stderr, "hello world! Yielding now...\n");
    // fprintf(stderr, "Waiting for input: ");
    // scanf("[^\n]");
    sched_yield();
    fprintf(stderr, "I'm back!\n");
    absl::SleepFor(absl::Milliseconds(10));
    fprintf(stderr, "fantastic nap!\n");
    // Verify that a ghost thread implicitly clones itself in the ghost
    // scheduling class.
    std::thread t2(
        [] { CHECK_EQ(sched_getscheduler(/*pid=*/0), SCHED_GHOST); });
    t2.join();
  });

  t.Join();
  printf("\nFinished simple worker\n");
}

}  // namespace
}  // namespace ghost

int main() {
  {
    printf("SimpleExp\n");
    ghost::ScopedTime time;
    ghost::SimpleExp();
  }
  return 0;
}

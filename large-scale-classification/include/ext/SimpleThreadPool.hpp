#ifndef SIMPLETHREADPOOL_H
#define SIMPLETHREADPOOL_H

#include <vector>
#include <functional>
#include <thread>
#include <shared_mutex>
#include <condition_variable>

#include "type.hpp"
#include "BlockingQueue.hpp"

namespace SimpleThreadPool {

  class ThreadPool {
  public:
    using mutex_type = std::mutex;
    using condition_variable_type = std::condition_variable;

  public:
    ThreadPool(size_type size) :
      m_threads(size), m_is_idle(size), m_stop_signal(false),
      m_task_queue(m_stop_signal) {
      std::fill(std::begin(m_is_idle), std::end(m_is_idle), true);
      for (index_type i = 0; i < m_threads.size(); i++) {
        m_threads[i] = std::thread(&ThreadPool::threadLoop, this, i);
      }
    }

    void putJob(const std::function<void()> &job) {
      m_task_queue.push(job);
    }

    void wait() {
      while (true) {
        std::unique_lock<mutex_type> lock(m_mutex);
        if (m_cond.wait_for(lock, std::chrono::milliseconds(500),
                            [this, &lock]() {
                              if (!m_task_queue.empty()) {
                                return false;
                              }
                              return checkAllIdle();
                            })) {
          break;
        }
      }
    }

    ~ThreadPool() {
      m_stop_signal = true;
      m_task_queue.notify();
      for (auto &&th : m_threads) {
        th.join();
      }
    }
  private:
    void threadLoop(index_type index) {
      while (!m_stop_signal) {
        std::function<void()> job;
        if (m_task_queue.pop(job, [this, index]() {
              m_is_idle[index] = false;
            })) {
          job();
          setSelfIdle(index);
        }
      }
    }

    void setSelfIdle(index_type index) {
      m_is_idle[index] = true;
      m_cond.notify_one();
    }

    bool checkAllIdle() {
      for (index_type i = 0; i < m_is_idle.size(); i++) {
        if (m_is_idle[i] == false) {
          return false;
        }
      }
      return true;
    }

  private:
    std::vector<std::thread> m_threads;
    std::vector<std::atomic_bool> m_is_idle;
    std::atomic_bool m_stop_signal;
    BlockingQueue< std::function<void()> > m_task_queue;

    mutex_type m_mutex;
    condition_variable_type m_cond;

  };


}  // SimpleThreadPool

#endif /* SIMPLETHREADPOOL_H */

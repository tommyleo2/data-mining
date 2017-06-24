#ifndef BlockingQueue_hpp
#define BlockingQueue_hpp

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <memory>

namespace SimpleThreadPool {

  template <typename T>
  class BlockingQueue {
  public:
    using container_type = std::queue<T>;
    using value_type = T;
    using mutex_type = std::mutex;
    using condition_variable_type = std::condition_variable;

  private:
    // Disallow copy constructor and assign operator
    BlockingQueue(const BlockingQueue &rhs) = delete;
    BlockingQueue &operator=(const BlockingQueue &rhs) = delete;

  public:
    BlockingQueue(std::atomic_bool &stop_signal) :
      m_stop_signal(stop_signal) { }

    bool pop(T &elem) {
      std::unique_lock<mutex_type> lock(m_mutex);
      m_cond.wait(lock, [this]() {
          return m_stop_signal == true || !m_queue.empty();
        });
      if (!m_queue.empty()) {
        elem = std::move(m_queue.front());
        m_queue.pop();
        notify();
        return true;
      }
      notify();
      return false;
    }

    bool pop(T &elem, const std::function<void()> &funcInLock) {
      std::unique_lock<mutex_type> lock(m_mutex);
      m_cond.wait(lock, [this]() {
          return m_stop_signal == true || !m_queue.empty();
        });
      if (!m_queue.empty()) {
        elem = std::move(m_queue.front());
        m_queue.pop();
        notify();
        funcInLock();
        return true;
      }
      notify();
      return false;
    }

    void push(const T &elem) {
      std::unique_lock<mutex_type> lock(m_mutex);
      m_queue.push(elem);
      notify();
    }

    bool push(T &&elem) {
      std::unique_lock<mutex_type> lock(m_mutex);
      m_queue.push(std::forward<T>(elem));
      notify();
      return true;
    }

    bool empty() const {
      std::unique_lock<mutex_type> lock(m_mutex);
      return m_queue.empty();
    }

    int size() const {
      std::unique_lock<mutex_type> lock(m_mutex);
      return m_queue.size();
    }

    void notify() { m_cond.notify_one(); }

  private:
    container_type m_queue;
    mutable mutex_type m_mutex;
    condition_variable_type m_cond;
    std::atomic_bool &m_stop_signal;
  };

}

#endif

#pragma once
#include <condition_variable>
#include <chrono>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <vector>

namespace utils {

using std::shared_ptr;
using std::vector;
using namespace std::chrono_literals;

template<typename T>
class TSQueue {
    int max_size_;
    std::mutex mutex_;
    std::condition_variable can_push_cond_;
    std::condition_variable can_pop_cond_;
    std::queue<T> queue_;
    volatile bool stopping_{false};

    typedef std::unique_lock<std::mutex> Lock;

public:
    explicit TSQueue(int max_size) : max_size_(max_size) {}

    bool push(const T& task, bool wait, bool notify = true) {
        Lock lock(mutex_);
        if (!wait && queue_.size() >= max_size_) {
            return false;
        }
        while (queue_.size() >= max_size_) {
            can_push_cond_.wait(lock);
        }
        queue_.push(task);
        if (notify)
            can_pop_cond_.notify_one();
        return true;
    }

    void emplace(T&& t) {
        Lock lock(mutex_);
        queue_.emplace(t);
        can_pop_cond_.notify_one();
    }

    T front() {
        Lock lock(mutex_);
        return queue_.empty() ? T() : queue_.front();
    }

    bool empty() {
        Lock lock(mutex_);
        return queue_.empty();
    }

    T pop(bool wait) {
        Lock lock(mutex_);
        while (wait && !stopping_ && queue_.empty()) {
            can_pop_cond_.wait(lock);
        }
        if (queue_.empty())
            return T();
        auto task = queue_.front();
        queue_.pop();
        can_push_cond_.notify_one();
        return task;
    }

    T pop_timeout(float timeout) {
        auto now = std::chrono::system_clock::now();
        Lock lock(mutex_);
        if (timeout <= 0) {
            can_pop_cond_.wait(lock, [this] { return !queue_.empty(); });
        } else {
            auto tp = now + int(timeout * 1000) * 1ms;
            if (!can_pop_cond_.wait_until(lock, tp, [this] { return !queue_.empty(); })) {
                throw std::runtime_error("Timeout");
            }
        }
        auto res = std::move(queue_.front());
        queue_.pop();
        return std::move(res);
    }

    vector<T> pop_multi(int limit, bool wait, int require, int max_token) {
        vector<T> tasks;
        int total_token_len = 0;
        {
            Lock lock(mutex_);
            while (wait && !stopping_ && queue_.size() < require) {
                can_pop_cond_.wait(lock);
            }
            while (!queue_.empty() && tasks.size() < limit) {
                total_token_len += queue_.front()->input_tokens.size();
                total_token_len += queue_.front()->bee_tokens.size();
                if (total_token_len >= max_token) {
                    break;
                }
                tasks.push_back(queue_.front());
                queue_.pop();
            }
        }
        if (!tasks.empty())
            can_push_cond_.notify_one();
        return tasks;
    }

    void stop() {
        Lock lock(mutex_);
        stopping_ = true;
        can_pop_cond_.notify_one();
    }

    size_t size() {
        Lock lock(mutex_);
        return queue_.size();
    }
};

}
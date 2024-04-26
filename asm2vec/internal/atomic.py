from typing import *
import threading


# 管理 threading.Lock 对象的获取和释放。
class LockContextManager:
    """A context manager for handling acquiring and releasing of threading locks."""
    def __init__(self, lock: threading.Lock):
        """
        Initialize the LockContextManager with a threading.Lock object.

        :param lock: The threading.Lock object to be managed.
        """
        self._lock = lock   # 要管理的锁对象
        self._exited = False   # 标记上下文管理器是否已经退出

    def __enter__(self):
        """Acquire the lock when entering the context."""
        self._lock.acquire()  # 进入上下文时获取锁

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Release the lock when exiting the context and set the _exited flag to True.

        :param exc_type: The type of exception that occurred.
        :param exc_val: The value of the exception that occurred.
        :param exc_tb: The traceback object of the exception that occurred.
        """
        self._exited = True  # 标记上下文管理器已经退出
        self._lock.release()  # 退出上下文时释放锁

    def exited(self) -> bool:
        """
        Check if the context manager has exited.

        :return: True if the context manager has exited, False otherwise.
        """
        return self._exited


# 提供一个线程安全的原子操作环境。
class Atomic:
    """A context manager for Atomic class to ensure thread-safe operations."""
    class AtomicContextManager(LockContextManager):
        def __init__(self, atomic: 'Atomic'):
            """
            Initialize the AtomicContextManager with an Atomic object.

            :param atomic: The Atomic object to be managed.
            """
            super().__init__(atomic._lock)  # 初始化基类的锁
            self._atomic = atomic  # 要包装的 Atomic 实例
            self._exited = False

        def __enter__(self):
            """Acquire the lock and return self when entering the context."""
            super().__enter__()  # 调用基类的 __enter__
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Release the lock when exiting the context."""
            super().__exit__(exc_type, exc_val, exc_tb)  # 调用基类的 __exit__

        def value(self) -> Any:
            """
            Get the value of the Atomic object.

            :return: The value of the Atomic object.
            :raises RuntimeError: If the context manager has exited.
            """
            if self.exited():
                raise RuntimeError('Trying to access AtomicContextManager after its exit.')
            return self._atomic._val  # 返回 Atomic 实例的值

        def set(self, value: Any) -> None:
            """
            Set the value of the Atomic object.

            :param value: The new value to be set.
            :raises RuntimeError: If the context manager has exited.
            """
            if self.exited():
                raise RuntimeError('Trying to access AtomicContextManager after its exit.')
            self._atomic._val = value  # 设置 Atomic 实例的值

    def __init__(self, value: Any):
        """
        Initialize the Atomic object with a value and a threading.Lock.

        :param value: The initial value to be stored in the Atomic object.
        """
        self._val = value  # Atomic 类的值
        self._lock = threading.Lock()   # Atomic 类的锁

    def lock(self) -> AtomicContextManager:
        """
        Create and return an AtomicContextManager instance for this Atomic object.

        :return: An AtomicContextManager instance.
        """
        return self.__class__.AtomicContextManager(self)  # 返回一个 AtomicContextManager 实例

    def value(self) -> Any:
        """
        Retrieve the value of the Atomic object in a thread-safe manner.

        :return: The value of the Atomic object.
        """
        with self.lock() as val:
            return val.value()

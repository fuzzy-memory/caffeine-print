import logging
from enum import Enum as PyEnum


class ChannelEnum(PyEnum):
    CURRENT_AFFAIRS = "current_affairs"
    BUSINESS = "business"


class TestingMode:
    _instance: "TestingMode" = None
    _is_testing: bool = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_testing_mode(self, is_testing: bool):
        self._is_testing = is_testing

    def get_testing_mode(self):
        return self._is_testing


class SelectedChannel:
    _instance: "SelectedChannel" = None
    _channel: ChannelEnum = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_channel(self, channel: ChannelEnum):
        self._channel = channel

    def get_channel(self):
        return self._channel

# -*- coding: utf-8 -*-
import os
import threading
from abc import abstractmethod

__author__ = "Romain Mormont <romainmormont@hotmail.com>"
__contributors__ = ["Ulysse Rubens <urubens@student.ulg.ac.be>"]
__version__ = "0.1"


class Logger(object):

    SILENT = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4

    def __init__(self, level, prefix=True):
        self._level = level
        self._prefix = prefix

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self._level = value

    def _log(self, level, message):
        if self._level >= level:
            formatted = self._format_msg(level, message)
            self._print(formatted)

    @abstractmethod
    def _print(self, message):
        pass

    def _format_msg(self, level, message):
        if self._prefix:
            rows = ["{} {}".format(self.prefix(level), row) for row in message.split(os.linesep)]
            return os.linesep.join(rows)
        else:
            return message

    def prefix(self, level):
        from datetime import datetime
        now = datetime.now().isoformat()
        tid = "{}".format(threading.current_thread().ident).zfill(6)
        fid = "tid:{}".format(tid)
        return "[{}][{}][{}]".format(fid, now, self.level2str(level))

    @classmethod
    def level2str(cls, level):
        if level == cls.DEBUG:
            return "DEBUG"
        elif level == cls.WARNING:
            return "WARN "
        elif level == cls.ERROR:
            return "ERROR"
        else:  # info
            return "INFO "

    def d(self, msg):
        """Alias for self.debug
        Parameters
        ----------
        msg: string
            The message to log
        """
        self.debug(msg)

    def debug(self, msg):
        """Logs a information message if the level of verbosity is above or equal DEBUG
        Parameters
        ----------
        msg: string
            The message to log
        """
        self._log(Logger.DEBUG, msg)

    def i(self, msg):
        """Alias for self.info
        Parameters
        ----------
        msg: string
            The message to log
        """
        self.info(msg)

    def info(self, msg):
        """Logs a information message if the level of verbosity is above or equal INFO
        Parameters
        ----------
        msg: string
            The message to log
        """
        self._log(Logger.INFO, msg)

    def w(self, msg):
        """Alias for self.warning
        Parameters
        ----------
        msg: string
            The message to log
        """
        self.warning(msg)

    def warning(self, msg):
        """Logs a information message if the level of verbosity is above or equal WARNING
        Parameters
        ----------
        msg: string
            The message to log
        """
        self._log(Logger.WARNING, msg)

    def e(self, msg):
        """Alias for self.error
        Parameters
        ----------
        msg: string
            The message to log
        """
        self.error(msg)

    def error(self, msg):
        """Logs a information message if the level of verbosity is above or equal ERROR
        Parameters
        ----------
        msg: string
            The message to log
        """
        self._log(Logger.ERROR, msg)


class StandardOutputLogger(Logger):
    """A logger printing the messages on the standard output
    """
    def _print(self, formatted_msg):
        print (formatted_msg)


class FileLogger(Logger):
    """A logger printing the messages into a file
    """
    def __init__(self, filepath, level, prefix=True):
        """Create a FileLogger
        Parameters
        ----------
        filepath: string
            Path to the logging file
        level: int
            Verbosity level
        prefix: bool
            True for adding a prefix for the logger
        """
        Logger.__init__(self, level, prefix=prefix)
        self._file = open(filepath, 'w+')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _print(self, formatted_msg):
        self._file.write(formatted_msg)

    def close(self):
        """Close the logging file
        """
        self._file.close()


class SilentLogger(Logger):
    """A logger that ignore messages
    """
    def __init__(self, prefix=True):
        Logger.__init__(self, Logger.SILENT, prefix=prefix)

    def _print(self, formatted_msg):
        pass

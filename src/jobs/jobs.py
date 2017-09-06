# -*- coding: utf-8 -*-
from abc import abstractmethod

from logger import StandardOutputLogger, Logger

__author__ = "Ulysse Rubens <urubens@student.ulg.ac.be>"
__version__ = "0.1"


class Job(object):
    def __init__(self, logger=StandardOutputLogger(Logger.INFO)):
        self.__logger = logger
        self.__job_done = False
        self.__progress = 0

    def done(self, status=True):
        self.__job_done = status

    def is_done(self):
        return self.__job_done

    def is_local(self):
        return False

    @property
    def logger(self):
        return self.__logger

    @logger.setter
    def logger(self, value):
        self.__logger = value

    def set_progress(self, status_comment=None, progress=None):
        if progress is not None and not 0 <= progress <= 100:
            raise ValueError("progress value should be between 0 and 100")

        if progress is not None:
            self.__progress = progress

        if status_comment is not None:
            self.__logger.info("{} ({}%)".format(status_comment, self.__progress))

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def close(self):
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        if value is None:
            # No exception, job is done
            self.done()
        self.close()
        return False


class LocalJob(Job):
    def is_local(self):
        return True

    def close(self):
        pass

    def start(self):
        pass


class CytomineJob(Job):
    def __init__(self, cytomine_client, software_id, project_id, parameters=None):
        super(CytomineJob, self).__init__()
        self.__cytomine = cytomine_client
        self.__software_id = software_id
        self.__project_id = project_id
        self.__job = None
        self.__parameters = parameters
        self.__progress = 0

    @property
    def cytomine_client(self):
        """
        Protected method

        Return
        ------
        cytomine : :class:`Cytomine`
            The Cytomine client
        """
        return self.__cytomine

    @property
    def project_id(self):
        """
        Protected method

        Return
        ------
        project_id : int
            The id of the project
        """
        return self.__project_id

    @property
    def software_id(self):
        """
        Protected method

        Return
        ------
        software_id : int
            The id of the software
        """
        return self.__software_id

    def start(self):
        """
        Connect to the Cytomine server and switch to job connection
        Incurs dataflows
        """
        current_user = self.__cytomine.get_current_user()
        if not current_user.algo:  # If user connects as a human (CLI execution)
            user_job = self.__cytomine.add_user_job(
                self.__software_id,
                self.__project_id
            )
            self.__cytomine.set_credentials(
                str(user_job.publicKey),
                str(user_job.privateKey)
            )
        else:  # If the user executes the job through the Cytomine interface
            user_job = current_user

        # set job state to RUNNING
        job = self.__cytomine.get_job(user_job.job)
        self.__job = self.__cytomine.update_job_status(job, status=job.RUNNING)

        # add software parameters
        if self.__parameters is not None:
            software = self.__cytomine.get_software(self.__software_id)
            self.__cytomine.add_job_parameters(self.__job.id, software, self.__parameters)

    def close(self):
        """
        Notify the Cytomine server of the job's end
        Incurs a dataflows
        """
        status = 4  # status code for FAILED
        if self.is_done():
            status = self.__job.TERMINATED

        self.__cytomine.update_job_status(self.__job, status=status)

    def set_progress(self, status_comment=None, progress=None):
        super(CytomineJob, self).set_progress(status_comment, progress)
        self.__job = self.__cytomine.update_job_status(self.__job,
                                                       status_comment=status_comment,
                                                       progress=progress)

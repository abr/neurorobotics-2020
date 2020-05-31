import os
import time
import warnings

import numpy as np
import h5py

class DataHandler():
    """
    Data handler for saving and loading data

    This class is meant to help simplify running automated tests. This is
    specifically helpful when running consecutive runs of learning where it is
    desired to pick up where the last run ended off.

    The naming convention used is as follows:
    - runs are consecutive tests, usually where the previous run is used as
    a starting point for the following run
    - a group of runs is called a session, multiple sessions are used for
      averaging and other statistical purposes like getting confidence
      intervals
    - a test_name is the user specified name of the test being run, which is
      made up of sessions that consist of several runs
    - by default the test_name data is saved in the abr_analyze database_dir.
      the location of this folder is specified in the path.py file

    However, this convention does not have to be used. Any save name can be
    passed in and data will be saved there, and can later be loaded.

    Parameters
    ----------
    db_name: string, Optional (Default: abr_analyze)
        name of the database being used
    """

    def __init__(self, db_dir=None, db_name='results'):
        if db_dir is None:
            db_dir = os.path.abspath('.')

        self.ERRORS = []
        self.db_loc = '%s/%s.h5'%(db_dir, db_name)
        # Instantiate the database object with the provided path so that it
        # gets created if it does not yet exist
        db = h5py.File(self.db_loc, 'a')
        # close the database after each function
        db.close()


    def save(self, data, save_location, overwrite=False, create=True,
             timestamp=True):
        """
        Saves the data dict passed in to the save_location specified in the
        instantiated database

        Parameters
        ----------
        data: dictionary of lists to save
            instantiate as
                data = {'data1': [], 'data2': []}
            append as
                data['data1'].append(np.copy(data_to_save))
                data['data2'].append(np.copy(other_data_to_save))
        save_location: string, Optional (Default: 'test')
            the group that all of the data will be saved to
        overwrite: boolean, Optional (Default: False)
            determines whether or not to overwrite data if group already exists
            An error gets triggered if the data is being saved to a group
            (folder) that already exists. Setting this to true will ignore that
            and save the data. Data will only get overwritten if the same key
            is used, otherwise the other data in the group will remain
            untouched
        create: boolean, Optional (Default: True)
            determines whether to create the group provided if it does not
            exist, or to warn to the user that it does not
        timestamp: boolean, Optional (Default: True)
            whether to save timestamp with data
        """

        if not isinstance(data, dict):
            raise TypeError('ERROR: data must be a dict, received ',
                            type(data))

        db = h5py.File(self.db_loc, 'a')
        if not self.check_group_exists(save_location):
            db.create_group(save_location)

        if timestamp:
            data['timestamp'] = time.strftime("%H:%M:%S")
            data['datestamp'] = time.strftime("%Y/%m/%d")

        for key in data:
            if key is not None:
                if data[key] is None:
                    data[key] = 'None'
                try:
                    try:
                        db[save_location].create_dataset(
                            '%s' % key, data=data[key])

                    except RuntimeError as e:
                        if overwrite:
                            # if dataset already exists, then overwrite data
                            del db[save_location+'/%s'%key]
                            db[save_location].create_dataset(
                                '%s' % key, data=data[key])
                        else:
                            print(e)
                            raise Exception(
                                'Dataset %s already exists in %s' %
                                (save_location, key) +
                                ': set overwrite=True to overwrite')
                except TypeError as type_error:
                    print('\n\n*****WARNING: SOME DATA DID NOT SAVE*****')
                    print('Trying to save %s to %s' % (key, save_location))
                    print('Received error: %s' %type_error)
                    print('NOTE: HDF5 has no None type and this dataHandler'
                          + ' currently has no test for None entries')
                    print('\n\n')

        db.close()


    def load(self, parameters, save_location):
        """
        Accepts a list of parameters and their path to where they are saved in
        the instantiated db, and returns a dictionary of the parameters and their
        values

        PARAMETERS
        ----------
        parameters: list of strings
            ex: ['q', 'dq', 'u', 'adapt']
            if you are unsure about what the keys are for the data saved, you
            can use the get_keys() function to list the keys in a provided
            group path
        save_location: string
            the location to look for data
            EX: 'test_group/test_name/session_num/run_num'
        """
        # check if the group exists
        exists = self.check_group_exists(location=save_location, create=False)

        # if group path does not exist, raise an exception to alert the user
        if exists is False:
            raise ValueError('The path %s does not exist'%(save_location))

        # otherwise load the keys
        db = h5py.File(self.db_loc, 'a')
        saved_data = {}
        for key in parameters:
            saved_data[key] = np.array(
                db.get('%s/%s' % (save_location, key)))

        db.close()

        return saved_data


    def delete(self, save_location):
        '''
        Deletes save_location and all contents from instantiated database

        PARAMETERS
        ----------
        save_location: string
            location in the instantiated database to delete
        '''
        #TODO: incoprorate KBHit to get user to verify deleting location and
        # print the keys so they are aware of what will be deleted
        try:
            db = h5py.File(self.db_loc, 'a')
            del db[save_location]
        except KeyError:
            warnings.warn('No entry for %s' % save_location)


    def rename(self, old_save_location, new_save_location, delete_old=True):
        '''
        Renames a group of dataset

        PARAMETERS
        ----------
        old_save_location: string
            save_location to dataset or group to be renamed
        new_save_location: string
            the new save_location to rename old_save_location as
        delete_old: Boolean, Optional(Default:True)
            True to delete old_save_location after renaming
            False to keep both the old and new save_locations
        '''
        db = h5py.File(self.db_loc, 'a')
        db[new_save_location] = db[old_save_location]
        if delete_old:
            del db[old_save_location]


    def get_keys(self, save_location):
        """
        Takes a path to an hdf5 dataset in the instantiated database and
        returns the keys at that location

        save_location: string
            save_location of the group that you want the keys from
            ex: 'my_feature_test/sub_test_group/session000/run003'
        """
        db = h5py.File(self.db_loc, 'a')
        if isinstance(db[save_location], h5py.Dataset):
            keys = [None]
        else:
            keys = list(db[save_location].keys())
        db.close()
        return keys


    def check_group_exists(self, location, create=False):
        """
        Accepts a location in the instantiated database and returns a boolean
        whether it exists. Additionally, the boolean create can be passed in
        that will create the group if it does not exist

        Parameters
        ----------
        location: string
            The database group that the function checks for,
        create: boolean, Optional (Default:True)
            true: create group if it does not exist
            false: do not create group if it does not exist
        """
        #TODO: should we add check if location is a dataset?
        db = h5py.File(self.db_loc, 'a')
        exists = location in db

        if exists is False:
            if create:
                db.create_group(location)
                exists = True
            else:
                exists = False
        db.close()

        return exists


    #TODO: make this function
    def sample_data(self):
        '''
        saves every nth data value to save on storage space
        '''
        raise Exception('This function is currently not supported')


    #NOTE: these are very control specific, should they be subclassed?
    #TODO: the following functions can probably be cleaned up and shortened
    def last_save_location(self, session=None, run=None, test_name='test',
                           test_group='test_group', create=True):
        """
        Following the naming structure of save_name/session(int)/run(int) for
        groups, the highest numbered run and session are returned (if not
        specified, otherwise the specified ones are checked)

        If the user sets session or run to None, the function searches the
        specified test_name for the highest numbered run in the
        highest numbered session, otherwise the specified values are used.
        Returns highest numbered run, session and path, unless a run or session
        was specified by the user to use.

        If the user specifies a session, or run that do not exist, the 0th
        session and/or run will be created. However, if the test_name does not
        exist, an exception will be raised to alert the user

        This function is used for both saving and loading, if used for saving
        then it may be desirable to create a group if the provided one does not
        exist. This can be done by setting create to True. The opposite is true
        for loading, where None should be returned if the group does not exist.
        In this scenario create should be set to False. By default, these are
        the settings for the load and save functions

        Parameters
        ----------
        session: int, Optional (Default: None)
            the session number of the current set of runs
            if set to None, then the latest session in the test_name folder
            will be use, based off the highest numbered session
        run: int, Optional (Default: None)
            the run number under which to save data
            if set to None, then the latest run in the test_name/session#
            folder will be used, based off the highest numbered run
        test_name: string, Optional (Default: 'test')
            the folder name that will hold the session and run folders
            the convention is abr_cache_folder/test_name/session#/run#
            The abr cache folder can be found in abr_control/utils/paths.py
        test_group: string, Optional (Default: 'test_group')
            the group that all of the various test_name tests belong to. This
            is helpful for grouping tests together relative to a specific
            feature being tested, a paper that they belong to etc.
        create: Boolean, Optional (Default: True)
            whether to create the group passed in if it does not exist
        """

        self.db = h5py.File(self.db_loc, 'a')

        # first check whether the test passed in exists
        exists = self.check_group_exists(
            location='%s/%s/' % (test_group, test_name), create=create)

        # if the test does not exist, return None
        if exists is False:
            run = None
            session = None
            location = '%s/%s/'%(test_group, test_name)
            self.db.close()
            return [run, session, location]

        # If a session is provided, check if it exists
        if session is not None:
            # check if the provided session exists before continuing, create it
            # if it does not and create is set to True
            exists = self.check_group_exists(
                location=(
                    '%s/%s/session%03d/' % (test_group, test_name, session)),
                create=create)
            # if exists, use the value
            if exists:
                session = 'session%03d' %session
            else:
                run = None
                session = None
                location = '%s/%s/' % (test_group, test_name)
                self.db.close()
                return [run, session, location]

        # if not looking for a specific session, check what our highest
        # numbered session is
        elif session is None:
            # get all of the session keys
            session_keys = list(
                self.db['%s/%s' % (test_group, test_name)].keys())

            if session_keys:
                session = max(session_keys)

            elif create:
                # No session can be found, create it if create is True
                self.db.create_group('%s/%s/session000' %
                                     (test_group, test_name))
                session = 'session000'

            else:
                run = None
                session = None
                location = '%s/%s/' % (test_group, test_name)
                self.db.close()
                return [run, session, location]

        if run is not None:
            # check if the provided run exists before continuing, create it
            # if it does not and create is set to True
            exists = self.check_group_exists(
                location='%s/%s/%s/run%03d' % (
                    test_group, test_name, session, run),
                create=create)
            # if exists, use the value
            if exists:
                run = 'run%03d' % run
            else:
                run = None
                location = '%s/%s/' % (test_group, test_name)
                self.db.close()
                return [run, session, location]

        # usually this will be set to None so that we can start from where we
        # left off in testing, but sometimes it is useful to pick up from
        # a specific run
        elif run is None:
            # get all of the run keys
            run_keys = list(self.db['%s/%s/%s' %
                                    (test_group, test_name, session)].keys())

            if run_keys:
                run = max(run_keys)

            else:
                run = None

        location = '%s/%s/' % (test_group, test_name)
        if session is not None:
            session = int(session[7:])
            location += 'session%03d/' % session
        else:
            location += '%s/' % session
        if run is not None:
            run = int(run[3:])
            location += 'run%03d' % run
        else:
            location += '%s/' % run

        self.db.close()
        return [run, session, location]


    def save_run_data(self, tracked_data, session=None, run=None,
                      test_name='test', test_group='test_group',
                      overwrite=False, create=True, timestamp=True):
        #TODO: currently module does not check whether a lower run or session
        # exists if the user provides a number for either parameter, could lead
        # to a case where user provides run to save as 6, but runs 0-5 do not
        # exist, is it worth adding a check for this?
        """ Saves data collected from test trials with
        standard naming convention.

        Uses the naming structure of a session being made up of several runs.
        This allows the user to automate scripts for saving and loading data
        between consecutive runs. These sets of runs are saved in a session, so
        multiple sessions can be run for averaging and other statistical
        purposes

        Parameters
        ----------
        tracked_data: dictionary of lists to save
            instantiate as
                tracked_data = {'data1': [], 'data2': []}
            append as
                tracked_data['data1'].append(np.copy(data_to_save))
                tracked_data['data2'].append(np.copy(other_data_to_save))
        session: int, Optional (Default: None)
            the session number of the current set of runs
            if set to None, then the latest session in the test_name folder
            will be use, based off the highest numbered session
        run: int, Optional (Default: None)
            the run number under which to save data
            if set to None, then the latest run in the test_name/session#
            folder will be used, based off the highest numbered run
        test_name: string, Optional (Default: 'test')
            the folder name that will hold the session and run folders
            the convention is abr_cache_folder/test_name/session#/run#
            The abr cache folder can be found in abr_control/utils/paths.py
        test_group: string, Optional (Default: 'test_group')
            the group that all of the various test_name tests belong to. This
            is helpful for grouping tests together relative to a specific
            feature being tested, a paper that they belong to etc.
        overwrite: boolean, Optional (Default: False)
            determines whether or not to overwrite data if a group / dataset
            already exists
        timestamp: boolean, Optional (Default: True)
            whether to save timestamp with data
        """

        if run is not None:
            run = 'run%03d' % run
        if session is not None:
            session = 'session%03d' % session
        if session is None or run is None:
            # user did not specify either run or session so we will grab the
            # last entry in the test_name directory based off the highest
            # numbered session and/or run
            [run, session, _] = self.last_save_location(
                session=session, run=run, test_name=test_name,
                test_group=test_group, create=create)

            # if no previous run saved, start saving in run0
            if run is None:
                run = 'run000'

        group_path = '%s/%s/%s/%s' % (test_group, test_name, session, run)

        # save the data
        self.save(data=tracked_data, save_location=group_path,
                  overwrite=overwrite, create=create, timestamp=timestamp)

    def load_run_data(self, parameters, session=None, run=None,
                      test_name='test', test_group='test_group', create=False):
        """
        Loads the data listed in parameters from the group provided

        The path to the group is used as 'test_group/test_name/session/run'
        Note that session and run are ints that from the user end, and are
        later used in the group path as ('run%i'%run) and ('session%i'%session)

        parameters: list of strings
            ex: ['q', 'dq', 'u', 'adapt']
            if you are unsure about what the keys are for the data saved, you
            can use the get_keys() function to list the keys in a provided
            group path
        session: int, Optional (Default: None)
            the session number of the current set of runs
            if set to None, then the latest session in the test_name folder
            will be use, based off the highest numbered session
        run: int, Optional (Default: None)
            the run number under which to save data
            if set to None, then the latest run in the test_name/session#
            folder will be used, based off the highest numbered run
        test_name: string, Optional (Default: 'test')
            the folder name that will hold the session and run folders
            the convention is abr_cache_folder/test_name/session#/run#
            The abr cache folder can be found in abr_control/utils/paths.py
        test_group: string, Optional (Default: 'test_group')
            the group that all of the various test_name tests belong to. This
            is helpful for grouping tests together relative to a specific
            feature being tested, a paper that they belong to etc.
        """
        # if the user doesn'r provide either run or session numbers, the
        # highest numbered run and session are searched for in the provided
        # test_group/test_name location
        if session is None or run is None:
            [run, session, group_path] = self.last_save_location(
                session=session, run=run, test_name=test_name,
                test_group=test_group, create=create)
        else:
            session = 'session%03d' % session
            run = 'run%03d' % run

        if run is None:
            saved_data = None
        else:
            group_path = '%s/%s/%s/%s' % (test_group, test_name, session, run)

            saved_data = self.load(parameters=parameters,
                                   save_location=group_path)

        return saved_data

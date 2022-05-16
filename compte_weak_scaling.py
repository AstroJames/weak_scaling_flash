#!/usr/bin/env python

"""
    Title: Turb.log analysis tool
    Author: James R. Beattie
    First Created: 28 Mar 2022

    A post-processing analysis tool for computing a weak scaling test of the FLASH simulation,
    time-stamped, and split into blocks between file I/O events. The time for file I/O events
    is also recorded.
    
    

    

"""

# Dependencies
################################################################################################

# Data handling
import os
import numpy as np
import argparse
import re
from datetime import datetime, timedelta
import pandas as pd

############################################################################################################
# Command Line Arguments
############################################################################################################

ap 			= argparse.ArgumentParser(description = 'Just a bunch of input arguments')
ap.add_argument('-f','--file',nargs="+",required=True,help='the list of files to create the plot from',type=str)
ap.add_argument('-d','--debug',action='store_true')
ap.add_argument('-file_writes','--file_writes',action='store_true')
ap.add_argument('-blocks','--blocks',action='store_true')
args 		= vars(ap.parse_args())

############################################################################################################
# Globals
############################################################################################################

seconds_per_hour = 3600

############################################################################################################
# Classes
############################################################################################################

class FLASHLogFileIO():
    def __init__(self,write_id):
        self.start_date = ""
        self.write_id = write_id
        self.wall_time = 0
        self.wall_time_norm = 0
        self.file_type = ""
        self.core_hours = 0
        self.open = False
        self.close = False

    def empty(self):
        self.start_date = ""
        self.wall_time = 0
        self.wall_time_norm = 0
        self.file_type = ""
        self.core_hours = 0
        self.open = False
        self.close = False

    def open_file(self):
        self.open = True
        if self.close:
            self.close = False

    def close_file(self):
        self.close = True
        if self.open:
            self.open = False

class FLASHLogBlock():
    def __init__(self,block_id):
        self.block_id = block_id
        self.core_hours = 0
        self.wall_time = 0
        self.avg_wall_time_diff = 0
        self.std_wall_time_diff = 0
        self.avg_wall_time_diff_norm = 0
        self.std_wall_time_diff_norm = 0
        self.dt = 0
        self.start_date = ""
        self.end_date = ""
        self.start_line = 0
        self.end_line = 0
        self.n_steps = 0

class FLASHLogStats:
    def __init__(self,file_path,debug=False,site="superMUC-NG"):
        self.turb_log_file_path = file_path
        self.site = site
        self.total_core_hours = 0
        if site == "superMUC-NG" or site == "Gadi":
            self.n_cores_per_node = 48
        self.nodes = None
        self.n_cores = None
        self.nxb = None
        self.nyb = None
        self.nzb = None
        self.n_cells_per_block = None
        self.n_cells = None
        self.blocks = {}
        self.fileIOs = {}
        self.debug = debug

    def search_core_count(self,line_number,line):
        search_obj = re.search(r"Number of MPI tasks:",line)
        if search_obj:
            match_obj = re.findall(r"\d+",line)
            self.n_cores = int(match_obj[0])
            self.nodes = self.n_cores / self.n_cores_per_node

    def search_block_cell_count(self,line_number,line,block_string):
        search_obj = re.search(block_string,line)
        if search_obj:
            match_obj = re.findall(r"\d+",line)
            block_cell_count = int(match_obj[0])
            return(block_cell_count)

    def search_walltime(self,line):
        # if the line contains a time step (n)
        search_obj = re.search(f"n=\d+",line)
        if search_obj: # if there is a code step in the line
            # extract the wall-time in H:M:S format
            wall_time_obj = re.findall(r"\d+\:\d+\:\d+\.\d+",line)
            date_obj = re.findall(r"\d+-\d+-\d+",line)
            #date = datetime.date(date_obj[0])
            wall_time = datetime.strptime(date_obj[0]+ " " + wall_time_obj[0], '%m-%d-%Y %H:%M:%S.%f')
            # extract the dt
            dt_obj = re.findall(r"dt=\d+.\d+\D+\d+",line)
            dt=float(dt_obj[0].split("=")[1])
            if self.debug:
                print(f"search_walltime: time={wall_time}, dt={dt}")
            return wall_time, dt
        else:  # if there is no code step, return none
            return None, None

    def search_file_write_stats(self,line,fileIO,IO_id_counter):
        # Extract the time and date and wall time
        wall_time_obj = re.findall(r"\d+\:\d+\:\d+\.\d+",line)
        date_obj = re.findall(r"\d+-\d+-\d+",line)
        wall_time = datetime.strptime(date_obj[0]+ " " + wall_time_obj[0], '%m-%d-%Y %H:%M:%S.%f')
        if re.search(f"open:",line):
            if fileIO.open:
                fileIO.empty()
            if self.debug:
                print("search_file_write_stats: found an open file")
            fileIO.open_file()
            fileIO.start_date = wall_time
            fileIO.wall_time = wall_time
            if self.debug:
                print(wall_time - fileIO.wall_time)
        elif re.search(f"close:",line):
            if self.debug:
                print("search_file_write_stats: closing file")
            fileIO.close_file()
            fileIO.wall_time = wall_time - fileIO.wall_time
            fileIO.wall_time = fileIO.wall_time.total_seconds()
            if self.debug:
                print(f"search_file_write_stats: {wall_time - fileIO.wall_time}")
            fileIO.core_hours = fileIO.wall_time / seconds_per_hour

    def compute_sim_parameters(self):
        if self.debug:
            print("compute_sim_parameters: first pass through the file to read sim. parameters.")
        with open(self.turb_log_file_path) as fp:
            line = fp.readline()
            cnt = 1
            while line:
                # get the number of cores
                if self.n_cores is None:
                    self.search_core_count(cnt,line)
                if self.nxb is None:
                    self.nxb = self.search_block_cell_count(cnt,line,"Number x zones:")
                if self.nyb is None:
                    self.nyb = self.search_block_cell_count(cnt,line,"Number y zones:")
                if self.nzb is None:
                    self.nzb = self.search_block_cell_count(cnt,line,"Number z zones:")
                line = fp.readline()
                cnt += 1

        self.n_cells_per_block = self.nxb * self.nyb * self.nzb
        self.n_cells = self.n_cells_per_block * self.n_cores

    def process_block_statistics(self,block,wall_time_list,cnt,dt_list,block_id_counter):
        # initilise a list for storing the steps in seconds
        steps_per_time_secs = []
        # now process all of the statistics for a single block
        block.end_date = wall_time_list[-1]
        block.end_line = cnt-1
        block.n_steps = block.end_line - block.start_line
        block.wall_time = wall_time_list
        block.dt = dt_list

        # time between steps
        steps_per_time = np.diff(block.wall_time)

        # slow loop (can't find a way to do this better with time.delta objects)
        # to convert time.delta objects into total seconds
        for idx, _ in enumerate(steps_per_time):
            steps_per_time_secs.append(steps_per_time[idx].total_seconds())

        # add to block attributes
        block.core_hours = (block.end_date - block.start_date) * self.n_cores
        block.avg_wall_time_diff = np.mean(steps_per_time_secs)
        block.std_wall_time_diff = np.std(steps_per_time_secs)
        block.avg_wall_time_norm = block.avg_wall_time_diff / self.n_cells_per_block
        block.std_wall_time_norm = block.std_wall_time_diff / self.n_cells_per_block

        # add the block to the FLASH class
        self.blocks[block_id_counter] = block

    def rename_and_transpose(self,data_frame):
        return data_frame.T.rename(columns={0:"entry_type",
                                            1:"id",
                                            2:"core_hrs",
                                            3:"n_steps",
                                            4:"avg_wall_time_per_step (s)",
                                            5:"avg_wall_time_norm (s)", # change this to normalised to wall time normalised
                                            6:"std_wall_time_norm (s)",
                                            7:"start_date"})

    def deconstruct_log_file(self):
        # local variables
        wall_time = None
        chunk = False
        write = False
        IO_id_counter = 0
        block_id_counter = 0
        wall_time_list = []
        dt_list = []

        # Make sure the simulation parameter objects are populated
        if self.n_cells is None:
            self.compute_sim_parameters()

        with open(self.turb_log_file_path) as fp: # with the log file open
            line = fp.readline()
            cnt = 1
            while line: # while there is a line to be read

                if args["file_writes"]:
                    if self.debug:
                        print("deconstruct_log_file: starting to initialise fileIO decomp.")
                    # search the line for check point of write plot files
                    search_chk = re.search("IO_writeCheckpoint",line)
                    search_plt = re.search("IO_writePlotfile",line)

                    # add the number of files IOs
                    # initialise a file
                    if search_chk is not None or search_plt is not None:
                        if write:
                            self.search_file_write_stats(line,fileIO,IO_id_counter)
                            if fileIO.close:
                                if search_chk is not None:
                                    fileIO.file_type = "chk"
                                elif search_plt is not None:
                                    fileIO.file_type = "plt"
                                fileIO.wall_time_norm = fileIO.wall_time / self.n_cells_per_block
                                write = False
                                self.fileIOs[IO_id_counter] = fileIO
                                IO_id_counter += 1
                        else:
                            write = True
                            fileIO = FLASHLogFileIO(IO_id_counter)
                            self.search_file_write_stats(line,fileIO,IO_id_counter)
                        if self.debug:
                            print(self.fileIOs)

                if args["blocks"]:
                    if self.debug:
                        print("deconstruct_log_file: starting to initialise block decomp.")
                    # search the line for the wall time
                    wall_time, dt = self.search_walltime(line)

                    # if there is a wall time in the search, i.e.,
                    # if we are inside of a chunk (or block) of time integrations
                    if wall_time is not None:
                        # if this is the first wall time in a chunk
                        if not chunk:
                            block = FLASHLogBlock(block_id_counter)
                            block.start_date = wall_time
                            block.start_line = cnt
                            chunk = True

                        # if the are inside of a block, then store the wall_time
                        # and dt
                        wall_time_list.append(wall_time)
                        dt_list.append(dt)

                    else: # if there is no wall time data
                        # if there is no wall time data AND we just finished a chunk
                        if chunk:
                            if self.debug:
                                print("deconstruct_log_file: entered a chunk")

                            # update all of the statistics of a block
                            self.process_block_statistics(block,wall_time_list,cnt,dt_list,block_id_counter)

                            # update / empty the counters / lists
                            block_id_counter += 1
                            wall_time_list = []
                            dt_list = []
                            chunk = False

                wall_time = None
                line = fp.readline()
                cnt += 1

    def create_dataset(self):
        if self.debug:
            print("create_block_dataset: creating the dataset")
        if args["blocks"]:
            number_of_blocks = len(self.blocks.keys())
            entry_type = np.repeat("block",number_of_blocks)
            block_data_frame = pd.concat(
                            [pd.DataFrame(
                            [entry_type[key],
                             self.blocks[key].block_id,
                             self.blocks[key].core_hours.total_seconds()/seconds_per_hour,
                             self.blocks[key].n_steps,
                             self.blocks[key].avg_wall_time_diff,
                             self.blocks[key].avg_wall_time_norm,
                             self.blocks[key].std_wall_time_norm,
                             self.blocks[key].start_date.strftime("%m-%d-%Y %H:%M:%S.%f")]) for key in self.blocks.keys()],
                            ignore_index=True,axis=1)
            block_data_frame = self.rename_and_transpose(block_data_frame)
            if not args["file_writes"]:
                block_data_frame.to_csv(f"{self.turb_log_file_path.split('.')[0]}_block_data.csv")

        if args["file_writes"]:
            number_of_fileIOs = len(self.fileIOs.keys())
            entry_type = np.repeat("fileIO",number_of_fileIOs)
            zero_fill  = np.zeros(number_of_fileIOs)
            fileIO_data_frame = pd.concat(
                            [pd.DataFrame(
                            [entry_type[key],
                             self.fileIOs[key].file_type,
                             self.fileIOs[key].core_hours,
                             zero_fill[key],
                             self.fileIOs[key].wall_time,
                             self.fileIOs[key].wall_time_norm,
                             zero_fill[key],
                             self.fileIOs[key].start_date.strftime("%m-%d-%Y %H:%M:%S.%f")]) for key in self.fileIOs.keys()],
                            ignore_index=True,axis=1)
            fileIO_data_frame = self.rename_and_transpose(fileIO_data_frame)
            if not args["blocks"]:
                fileIO_data_frame.to_csv(f"{self.turb_log_file_path.split('.')[0]}_fileIO_data.csv")

        if args["file_writes"] and args["blocks"]:
            data_frame = pd.concat([block_data_frame,fileIO_data_frame])
            data_frame.to_csv(f"{self.turb_log_file_path.split('.')[0]}_log_data.csv")

def main():
    for file in args["file"]:
        print(f"Beginning on file: {file}")
        turb_log = FLASHLogStats(file,args["debug"])
        turb_log.compute_sim_parameters()
        turb_log.deconstruct_log_file()
        turb_log.create_dataset()

############################################################################################################
# Main
############################################################################################################

if __name__ == "__main__":
    main()

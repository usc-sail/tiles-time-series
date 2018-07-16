import os

def get_fitbit_data_folder(main_data_folder):
    return os.path.join(main_data_folder, 'keck_wave1/2_preprocessed_data/fitbit')

def get_omsignal_data_folder(main_data_folder):
    return os.path.join(main_data_folder, 'keck_wave1/2_preprocessed_data/omsignal')

def get_ground_truth_folder(main_data_folder):
    return os.path.join(main_data_folder, 'keck_wave1/2_preprocessed_data/ground_truth')

def get_job_shift_folder(main_data_folder):
    return os.path.join(main_data_folder, 'keck_wave1/2_preprocessed_data/job shift')

def get_sleep_routine_work_folder(output_folder):
    return os.path.join(output_folder, 'sleep_routine_work')

def get_om_signal_start_end_recording_folder(output_folder):
    return os.path.join(output_folder, 'om_signal_timeline')

def get_sleep_timeline_data_folder(output_folder):
    return os.path.join(output_folder, 'sleep_timeline')
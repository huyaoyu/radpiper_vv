#! /usr/bin/python

# This file is for writing out spectra that are in the output of postproc.py
#  If you are running with run_postprocess_batch.py, like this:
#  python -i run_postprocess_batch.py baglist.csv /directory/where/bags/are/ True
#  After run_postprocess_batch.py completes it will return to the Python 
#  command prompt. Now you can run:
#  import write_out_spectra
#  write_out_spectra.write_full_pipe_spectrum('my_filename.csv',postProcessOutputArray[0])
#  write_out_spectra.write_segment_spectra('my_filetag',postProcessOutputArray[0])
#
#  This should result in the following files being written: 
#  my_filename.csv, my_filetag_forward.csv, my_filetag_reverse.csv

# Write the spectrum in postProcessOutput['full_pipe_spectrum'] to the file given
#    by filename as a single-line comma-separated list
def write_full_pipe_spectrum(filename,postProcessOutput):
    if 'full_pipe_spectrum' in postProcessOutput.keys():
        try:
            fh = open(filename,'w')
            for i in range(len(postProcessOutput['full_pipe_spectrum'])):
                fh.write('%f,' % postProcessOutput['full_pipe_spectrum'][i])
            fh.close()
        except IOError:
            print('Error! Could not write full pipe spectrum to file: '+filename)
    else:
        print('Error! Key full_pipe_spectrum does not exist in postProcessOutput object given as argument.')

# Write the spectra in postProcessOutput['items'][i]['item_forward_spectrum'] and 
#    postProcessOutput['items'][i]['item_backward_spectrum'] to files given by
#    filetag+'_forward.csv' and filetag+'_reverse.csv', respectively. Format of
#    files will be one row per segment, first column is collection time,
#    remaining comma-separated columns are spectrum values
def write_segment_spectra(filetag,postProcessOutput):
    if len(postProcessOutput['items']) > 0:
        try:
            fh_f = open(filetag+'_forward.csv','w')
            fh_r = open(filetag+'_reverse.csv','w')
            for i in range(len(postProcessOutput['items'])):
                fh_f.write('%f,' % postProcessOutput['items'][i]['item_forward_live_time'])
                for j in range(len(postProcessOutput['items'][i]['item_forward_spectrum'])):
                    fh_f.write('%f,' % postProcessOutput['items'][i]['item_forward_spectrum'][j])
                fh_f.write('\n')
                fh_r.write('%f,' % postProcessOutput['items'][i]['item_backward_live_time'])
                for j in range(len(postProcessOutput['items'][i]['item_backward_spectrum'])):
                    fh_r.write('%f,' % postProcessOutput['items'][i]['item_backward_spectrum'][j])
                fh_r.write('\n')
            fh_f.close()
            fh_r.close()
        except IOError:
            print('Error! Could not write segment spectra to files.')
    else:
        print("No items in postProcessOutput['items'] array. Exiting.")

# Write out results from full pipe check, including whether full pipe check flag was thrown
def write_full_pipe_check_results(filename,postProcessOutput):
    if 'full_pipe_peak_channel' in postProcessOutput.keys():
        try:
            fh = open(filename,'w')
            fh.write('full_pipe_peak_channel = %f\n' % postProcessOutput['full_pipe_peak_channel'])
            fh.write('full_pipe_peak_fwhm = %f\n' % postProcessOutput['full_pipe_peak_fwhm'])
            if postProcessOutput['batch_flag'].find('full_spectrum_check_fail') != -1:
               fh.write('Flagged full pipe? True')
            else:
               fh.write('Flagged full pipe? False')
            fh.close()
        except IOError:
            print('Error! Unable to write results to file: '+filename)
    else:
        print('Error! Key full_pipe_peak_channel does not exist in postProcessOutput')

# Write out results from contamination check
def write_contamination_check_results(filename,postProcessOutput):
    try:
        fh = open(filename,'w')
        fh.write('con_cps_start = %f\n' % postProcessOutput['con_cps_start'])
        fh.write('con_cps_start_sigma = %f\n' % postProcessOutput['con_cps_start_sigma'])
        fh.write('con_cps_start_mda = %f\n' % postProcessOutput['con_cps_start_mda'][0])
        fh.write('con_cps_end = %f\n' % postProcessOutput['con_cps_end'])
        fh.write('con_cps_end_sigma = %f\n' % postProcessOutput['con_cps_end_sigma'])
        fh.write('con_cps_end_mda = %f\n' % postProcessOutput['con_cps_end_mda'][0])
        fh.write('con_passfail = %s\n' % postProcessOutput['con_passfail'])
        fh.close()
    except IOError:
        print('Error! Unable to write results to file: '+filename) 


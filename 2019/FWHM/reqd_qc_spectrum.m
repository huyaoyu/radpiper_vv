function [counts] = reqd_qc_spectrum(fn, col)
% Read the qc_spectrum.csv file. Return the values of the "counts" column.
% There should be two rows of data and one header row.
% fn: The file name with full path.
% col: The column index of "counts".
%

% Open the file.
fp = fopen(fn, 'r');

s = fgetl(fp);

lineCount = 1;

while s ~= -1
    lineCount = lineCount + 1;
    s = fgetl(fp);
    
    % Split.
    
    
    
    
end

flose(fp);


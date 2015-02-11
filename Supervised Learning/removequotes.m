function ret = removequotes( x )
%REMOVEQUOTES This function removes any double quotes in a character
%string.
%   This function is written to complement a call from datasetfun to remove
%   unnecessary double quotes from the bank dataset.

% Copyright 2013 The MathWorks, Inc.

if isa(x,'cell')
    ret = regexprep(x,'"','');
elseif isa(x,'nominal')
    ret = regexprep(cellstr(x),'"','');
else
    ret = x;
end
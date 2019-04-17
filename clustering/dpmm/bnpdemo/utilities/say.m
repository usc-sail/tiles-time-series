function say(varargin);
% print statement to screen.
% if an equal sign is encountered the following expression is evaluated
% and output printed.
% to print a string including spaces put them in single quotes.
% to print an equal sign use backslash i.e. \=
% to print comma, semicolon, parentheses or braces use quotes e.g. ',' ';'
% to not print space between strings place a \ at end of preceding string.
% if an exclamation mark is encountered the following expression is evaluated,
% if this is > verboselevel (global variable) rest of expressions not printed.
% Examples:
%    say one two three                          %> one two three
%    say one plus three \= =1+3                 %> one plus three = 4
%    a = 3; say a+a \= =a+a                     %> a+a = 6
%    say 'C{\' =a\ ','\ =a+a\ '};'              %> C{3,6};
%    global verboselevel; verboselevel = 3      % by default is 0
%    say one !1 two !4 three                    %> one two
%    say one !4 two !1 three                    %> one
%
% Note: "say =a+a" does not work as matlab interprets this as assigning a+a to
% variable "say".

for i=1:nargin
  if ischar(varargin{i}) 
    if varargin{i}(end)=='\'
      format = '%s';
      varargin{i}(end) = '';
    else
      format = '%s ';
    end
    if varargin{i}(1)=='='
      varargin{i} = evalin('caller',varargin{i}(2:end));
    elseif varargin{i}(1)=='\'
      varargin{i} = varargin{i}(2:end);
    elseif varargin{i}(1)=='!' % verbosity
      global verboselevel
      if ~exist('verboselevel'), verboselevel = 0; end
      level = evalin('caller',varargin{i}(2:end));
      if level > verboselevel, fprintf(1,'\n'); return; end
      continue;
    end
  end
  if ischar(varargin{i})
    fprintf(1,format,varargin{i});
  elseif isnumeric(varargin{i})
    fprintf(1,format,num2str(varargin{i}));
  else
    disp(varargin{i});
  end

end
fprintf(1,'\n');


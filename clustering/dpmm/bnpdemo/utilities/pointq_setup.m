function pointq_setup(hh);
%    function pointq_setup(hh)
% Sets up a points queue.  Everytime a user clicks in the axes (given by hh)
% the point is added to the UserData variable associated with the axes.
% the UserData variable is n*2, where n is the number of points in the queue.

set(hh,'userdata',zeros(0,2));
set(hh,'buttondownfcn',@add2q);
set(hh,'interruptible','off');
set(hh,'busyaction','queue');

function add2q(hh,event)

pointerline = get(hh,'currentpoint');
pointerlocation = pointerline(1,1:2);

qq = get(hh,'userdata');
qq(end+1,:) = pointerlocation;
set(hh,'userdata',qq);

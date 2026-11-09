import sys

if len(sys.argv) > 1 and sys.argv[1] == 'control':
    from .control import main
    raise SystemExit(main(sys.argv[2:]))

if len(sys.argv) > 1 and sys.argv[1] == 'ctl':
    from .ctl import main
    raise SystemExit(main(sys.argv[2:]))

from .controller import main

raise SystemExit(main())

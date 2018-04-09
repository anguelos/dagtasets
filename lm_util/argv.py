import sys

def get_arg_switches(default_switches, argv=None):
    default_switches = dict(default_switches, **{"help": False})
    if argv is None:
        argv = sys.argv
    argv_switches = dict(default_switches)
    argv_switches.update([(arg[1:].split("=") if "=" in arg else [arg[1:], "True"]) for arg in argv if arg[0] == "-"])
    if set(argv_switches.keys()) > set(default_switches.keys()):
        help_str = "\n" + argv[0] + " Syntax:\n\n"
        for k in default_switches.keys():
            help_str += "\t-%s=%s. Default %s\n" % (
                k, repr(type(default_switches[k])), repr(default_switches[k]))
        help_str += "\n\nUrecognized switches: "+repr(tuple( set(default_switches.keys()) - set(argv_switches.keys())))
        help_str += "\nAborting.\n"
        sys.stderr.write(help_str)
        sys.exit(1)

    # Setting argv element to the value type of the default.
    argv_switches.update({k: type(default_switches[k])(argv_switches[k]) for k in argv_switches.keys() if type(default_switches[k]) != str and type(argv_switches[k]) == str})

    positionals = [arg for arg in argv if arg[0] != "-"]
    argv[:] = positionals

    help_str = "\n" + argv[0] + " Syntax:\n\n"

    for k in default_switches.keys():
        help_str += "\t-%s=%s. Default %s . Passed %s\n" % (
        k, repr(type(default_switches[k])), repr(default_switches[k]), repr(argv_switches[k]))
    help_str += "\nAborting.\n"

    if argv_switches["help"]:
        sys.stderr.write(help_str)
        sys.exit()
    del argv_switches["help"]

    return argv_switches, help_str

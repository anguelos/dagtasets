import sys
import re

def get_arg_switches(default_switches, argv=None):
    new_default_switches={}
    switches_help = {"help":"Print help and exit."}
    for k,v in default_switches.items():
        if  hasattr(v, '__len__') and len(v)==2 and isinstance(v[1], basestring):
            switches_help[k]=v[1]
            new_default_switches[k]=v[0]
        else:
            switches_help[k] = ""
            new_default_switches[k] = v
    default_switches=new_default_switches
    del new_default_switches


    default_switches = dict(default_switches, **{"help": False})
    if argv is None:
        argv = sys.argv
    argv_switches = dict(default_switches)
    argv_switches.update([(arg[1:].split("=") if "=" in arg else [arg[1:], "True"]) for arg in argv if arg[0] == "-"])
    if set(argv_switches.keys()) > set(default_switches.keys()):
        help_str = "\n" + argv[0] + " Syntax:\n\n"
        for k in default_switches.keys():
            help_str += "\t-%s=%s %s Default %s.\n" % (
                k, repr(type(default_switches[k])), switches_help[k], repr(default_switches[k]))
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
        help_str += "\t-%s=%s %s Default %s . Passed %s\n" % (
        k, repr(type(default_switches[k])), switches_help[k], repr(default_switches[k]), repr(argv_switches[k]))
    help_str += "\nAborting.\n"

    #replace {blabla} with argv_switches["balbla"] values
    replacable_values=["{"+k+"}" for k in argv_switches.keys()]
    while len(re.findall("{[a-z0-9A-Z_]+}","".join([v for v in argv_switches.values() if isinstance(v,str)]))):
        for k,v in argv_switches.items():
            if isinstance(v,str):
                argv_switches[k]=v.format(**argv_switches)

    if argv_switches["help"]:
        sys.stderr.write(help_str)
        sys.exit()
    del argv_switches["help"]

    return argv_switches, help_str

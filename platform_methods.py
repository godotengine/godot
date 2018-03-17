import os
import sys
import json
import uuid
import functools
import subprocess

# NOTE: The multiprocessing module is not compatible with SCons due to conflict on cPickle


def run_in_subprocess(builder_function):

    @functools.wraps(builder_function)
    def wrapper(target, source, env):

        # Convert SCons Node instances to absolute paths
        target = [node.srcnode().abspath for node in target]
        source = [node.srcnode().abspath for node in source]

        # Short circuit on non-Windows platforms
        if os.name != 'nt':
            return builder_function(target, source, None)

        # Identify module
        module_name = builder_function.__module__
        function_name = builder_function.__name__
        module_path = sys.modules[module_name].__file__
        if module_path.endswith('.pyc') or module_path.endswith('.pyo'):
            module_path = module_path[:-1]

        # Subprocess environment
        subprocess_env = os.environ.copy()
        subprocess_env['PYTHONPATH'] = os.pathsep.join([os.getcwd()] + sys.path)

        # Save parameters
        args = (target, source, None)
        data = dict(fn=function_name, args=args)
        json_path = os.path.join(os.environ['TMP'], uuid.uuid4().hex + '.json')
        with open(json_path, 'wt') as json_file:
            json.dump(data, json_file, indent=2)
        try:
            print('Executing builder function in subprocess: module_path=%r; data=%r' % (module_path, data))
            exit_code = subprocess.call([sys.executable, module_path, json_path], env=subprocess_env)
        finally:
            try:
                os.remove(json_path)
            except (OSError, IOError) as e:
                # Do not fail the entire build if it cannot delete a temporary file
                print('WARNING: Could not delete temporary file: path=%r; [%s] %s' %
                      (json_path, e.__class__.__name__, e))

        # Must succeed
        if exit_code:
            raise RuntimeError(
                'Failed to run builder function in subprocess: module_path=%r; data=%r' % (module_path, data))

    return wrapper


def subprocess_main(namespace):

    with open(sys.argv[1]) as json_file:
        data = json.load(json_file)

    fn = namespace[data['fn']]
    fn(*data['args'])

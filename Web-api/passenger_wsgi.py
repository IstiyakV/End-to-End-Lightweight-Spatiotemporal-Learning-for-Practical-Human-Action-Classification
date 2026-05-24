import os
import sys
import traceback

# Force PyTorch and underlying linear algebra libraries to use single-threaded execution for compliance
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Add application directory to sys.path to ensure self-contained imports when hosted
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Programmatically detect and activate cPanel/Namecheap virtual environment paths to prevent ImportErrors
    python_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    if 'virtualenv' in sys.executable:
        venv_base = os.path.dirname(os.path.dirname(sys.executable))
        site_packages = os.path.join(venv_base, 'lib', python_ver, 'site-packages')
        if os.path.exists(site_packages) and site_packages not in sys.path:
            sys.path.insert(0, site_packages)

    # Lightning-fast, non-recursive cPanel virtualenv site-packages scanner (takes under 1ms!)
    virtualenv_root = '/home/istyeyco/virtualenv'
    if os.path.exists(virtualenv_root):
        for app_name in os.listdir(virtualenv_root):
            app_path = os.path.join(virtualenv_root, app_name)
            if os.path.isdir(app_path):
                lib_path = os.path.join(app_path, 'lib')
                if os.path.exists(lib_path):
                    for py_ver in os.listdir(lib_path):
                        site_packages = os.path.join(lib_path, py_ver, 'site-packages')
                        if os.path.exists(site_packages) and site_packages not in sys.path:
                            sys.path.insert(0, site_packages)

    app_root = os.path.dirname(os.path.abspath(__file__))
    for venv_name in ['venv', '.venv', 'virtualenv']:
        site_packages = os.path.join(app_root, venv_name, 'lib', python_ver, 'site-packages')
        if os.path.exists(site_packages) and site_packages not in sys.path:
            sys.path.insert(0, site_packages)

    # Import the Flask application object as 'application' (demanded by cPanel Passenger WSGI)
    from app import app as application
except Exception as e:
    # Catch any startup/import errors and expose the traceback as a clean HTTP 200 response for debugging
    tb_str = traceback.format_exc()
    def application(environ, start_response):
        start_response('200 OK', [('Content-Type', 'text/plain')])
        debug_output = (
            f"LUMAT WSGI Initialization Failed!\n\n"
            f"Traceback:\n{tb_str}\n"
            f"sys.path:\n" + "\n".join(sys.path) + "\n\n"
            f"sys.executable:\n{sys.executable}\n"
        )
        return [debug_output.encode('utf-8')]

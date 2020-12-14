__all__ = ['__version__']

# major, minor, patch
version_info = [1, 1, 0]

# suffix
suffix = None

__version__ = '.'.join(map(str, version_info)) + (f'.{suffix}' if suffix else '')

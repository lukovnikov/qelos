from torch import nn
import random


class ModuleList(nn.Module):
    """Holds submodules in a list.
    ModuleList can be indexed like a regular Python list, but modules it
    contains are properly registered, and will be visible by all Module methods.
    Arguments:
        modules (list, optional): a list of modules to add
    Example::
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        self._ordering = []
        if modules is not None:
            self += modules

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return self._modules[self._ordering[idx]]

    def __setitem__(self, idx, module):
        return setattr(self, self._ordering[idx], module)

    def __delitem__(self, idx):
        delattr(self, self._ordering[idx])
        del self._ordering[idx]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        for x in self._ordering:
            yield self._modules[x]

    def __iadd__(self, modules):
        return self.extend(modules)

    def append(self, module):
        """Appends a given module at the end of the list.
        Arguments:
            module (nn.Module): module to append
        """
        newhash = self._make_new_hash()
        self._ordering.append(newhash)
        self.add_module(newhash, module)
        return self

    def insert(self, i, module):
        """ Inserts a given module at a given position.
        Arguments:
            i (int): position where to insert
            module (nn.Module): module to insert
        """
        newhash = self._make_new_hash()
        self._ordering.insert(i, newhash)
        self.add_module(newhash, module)
        return self

    def extend(self, modules):
        """Appends modules from a Python list at the end.
        Arguments:
            modules (list): list of modules to append
        """
        if not isinstance(modules, list):
            raise TypeError("ModuleList.extend should be called with a "
                            "list, but got " + type(modules).__name__)
        for i, module in enumerate(modules):
            newhash = self._make_new_hash()
            self._ordering.append(newhash)
            self.add_module(newhash, module)
        return self

    def _make_new_hash(self):
        newhash = str(random.getrandbits(32))
        while newhash in self._modules:
            newhash = str(random.getrandbits(32))
        return newhash
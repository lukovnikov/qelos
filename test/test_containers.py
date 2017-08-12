from unittest import TestCase
from torch import nn
from qelos.containers import ModuleList


class TestModuleList(TestCase):
    def test_ListModule(self):
        modules = [nn.ReLU(), nn.Linear(5, 5)]
        module_list = ModuleList(modules)

        def check():
            self.assertEqual(len(module_list), len(modules))
            for m1, m2 in zip(modules, module_list):
                self.assertIs(m1, m2)
            children = list(module_list.children())
            for m1 in modules:      # every module is a child of module_list
                self.assertIn(m1, children)
            for m2 in children:     # every child is in modules
                self.assertIn(m2, modules)
            #for m1, m2 in zip(modules, module_list.children()):     # do the children also have to be ordered?
            #    self.assertIs(m1, m2)
            for i in range(len(modules)):
                self.assertIs(module_list[i], modules[i])

        check()
        modules += [nn.Conv2d(3, 4, 3)]
        module_list += [modules[-1]]
        check()
        modules.append(nn.Tanh())
        module_list.append(modules[-1])
        check()
        next_modules = [nn.Linear(5, 5), nn.Sigmoid()]
        modules.extend(next_modules)
        module_list.extend(next_modules)
        check()
        modules[2] = nn.Conv2d(5, 3, 2)
        module_list[2] = modules[2]
        check()
        modules.insert(2, nn.SELU())
        module_list.insert(2, modules[2])
        check()
        del modules[1]
        del module_list[1]
        check()

        with self.assertRaises(TypeError):
            module_list += nn.ReLU()
        with self.assertRaises(TypeError):
            module_list.extend(nn.ReLU())

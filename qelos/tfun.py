import qelos as q
import torch
import numpy as np
import tensorflow as tf


class TFfunction(torch.autograd.Function):
    def __init__(self, forward_graph, backward_graph, session,
                 *x, **kw):
        super(TFfunction, self).__init__(*x, **kw)
        self.__inputs, self.__outputs = forward_graph
        self.__output_grads, self.__input_grads = backward_graph
        self.tfsess = session      # TODO: closing session

    def forward(self, *x):
        self.save_for_backward(*x)
        _tf_out = self.tfsess.run(self.__outputs,
                                  feed_dict={
                self.__inputs: [xe.cpu().numpy() for xe in x],
                                            })
        if q.issequence(_tf_out):
            return tuple([torch.from_numpy(_tf_out_e) for _tf_out_e in _tf_out])
        else:
            return torch.from_numpy(_tf_out)
        # if not q.issequence(_tf_out):
        #     _tf_out = (_tf_out,)
        # _tf_out = tuple([torch.from_numpy(_tf_out_e) for _tf_out_e in _tf_out])
        # if len(_tf_out) == 1:
        #     _tf_out = _tf_out[0]
        # return _tf_out

    def backward(self, *y_grad):
        saved_inputs = self.saved_tensors
        # saved_inputs = map(lambda x: x.numpy(), saved_inputs)
        saved_inputs = [saved_input.numpy() for saved_input in saved_inputs]
        if q.issequence(y_grad):
            y_grad = tuple(map(lambda y_grad_e: y_grad_e.clone().cpu().numpy(), y_grad))
        else:
            y_grad = y_grad.clone().cpu().numpy()
        print(type(y_grad))
        all_grads = self.tfsess.run(self.__input_grads,
                                 feed_dict={self.__inputs: saved_inputs,
                                            self.__output_grads: y_grad})

        all_grads = tuple([copy_from_numpy(x_grad_e) for x_grad_e in all_grads])
        x_grad = all_grads
        return x_grad


def copy_from_numpy(x):
    ret = torch.zeros(x.shape)
    retsize = ret.size()
    flatret = ret.view(-1)
    flatx = x.flatten()
    for i in range(len(flatx)):
        flatret[i] = float(flatx[i])
    ret = flatret.view(retsize)
    return ret


class TFModule(torch.nn.Module):
    def __init__(self, **kw):
        super(TFModule, self).__init__(**kw)
        self.__dummyparam = torch.nn.Parameter(torch.zeros(1))
        self.__forward_graph = None
        self.__backward_graph = None
        self._params = torch.nn.ParameterList()
        self.__tfparam2param = {}
        self.__param2tfparam = {}
        self.__tfsess = tf.Session()

    def _build_forward_graph(self, *x):
        tfinps = create_placeholders_from_variables(*x)
        tfouts = self.apply(*tfinps)
        self.__tfsess.run(tf.global_variables_initializer())
        tfparams = tf.trainable_variables()
        _tfparams = []
        for tfparam in tfparams:
            paramval = self.__tfsess.run(tfparam)
            param = q.var(paramval).cuda(self.__dummyparam).v
            param = torch.nn.Parameter(param.data)
            self._params.append(param)
            self.__tfparam2param[tfparam.name] = param
            self.__param2tfparam[param] = tfparam
            _tfparams.append(tfparam)
        self.__forward_graph = (tfinps + tuple(_tfparams), tfouts)

    def _build_backward_graph(self):
        if self.__forward_graph is None:
            self._build_forward_graph()
        forward_inps, forward_outs = self.__forward_graph
        # make a grad placeholder for each forward out
        if q.issequence(forward_outs):
            forward_outs = forward_outs
            forward_outs_grads = tuple([tf.placeholder(forward_out.dtype, shape=forward_out.shape) for forward_out in forward_outs])
        else:
            forward_outs_grads = tf.placeholder(forward_outs.dtype, shape=forward_outs.shape)
        # make grads
        backward_outs = tf.gradients(forward_outs, forward_inps, grad_ys=forward_outs_grads)
        self.__backward_graph = (forward_outs_grads, backward_outs)

    def forward(self, *x):
        if self.__forward_graph is None:
            self._build_forward_graph(*x)
        if self.__backward_graph is None:
            self._build_backward_graph()
        fun = TFfunction(self.__forward_graph,
                         self.__backward_graph,
                         self.__tfsess)
        params = self._params
        params = [param for param in params]
        out = fun(*(x + tuple(params)))
        return out

    def apply(self, *x):         # x is a tensorflow placeholder
        raise NotImplemented()


class DummyTFModule(TFModule):
    def apply(self, x):
        return x + 1


class DummyTFModule2(TFModule):
    def apply(self, x, a):
        """ in tensorflow domain """
        y = x + a
        param = tf.get_variable("tf_param", [4, 3])
        z = y * param
        return z, y


def create_placeholders_from_variables(*x):
    typemap = {torch.FloatTensor: tf.float32}
    tfxs = [tf.placeholder(typemap[type(xe.data)], shape=xe.size()) for xe in x]
    return tuple(tfxs)


def test():
    from scipy.signal import convolve2d, correlate2d
    from torch.nn.modules.module import Module
    from torch.nn.parameter import Parameter

    class ScipyConv2dFunction(torch.autograd.Function):
        def forward(self, input, filter):
            result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
            self.save_for_backward(input, filter)
            return torch.FloatTensor(result)

        def backward(self, grad_output):
            input, filter = self.saved_tensors
            grad_input = convolve2d(grad_output.numpy(), filter.t().numpy(), mode='full')
            grad_filter = convolve2d(input.numpy(), grad_output.numpy(), mode='valid')
            return torch.FloatTensor(grad_input), torch.FloatTensor(grad_filter)

    class ScipyConv2d(Module):
        def __init__(self, kh, kw):
            super(ScipyConv2d, self).__init__()
            self.filter = Parameter(torch.randn(kh, kw))

        def forward(self, input):
            return ScipyConv2dFunction()(input, self.filter)

    module = ScipyConv2d(3, 3)
    print(list(module.parameters()))
    input = torch.autograd.Variable(torch.randn(10, 10), requires_grad=True)
    output = module(input)
    print(output)
    output.backward(torch.randn(8, 8))
    print(input.grad)


if __name__ == "__main__":
    # test()
    # print("test done")

    x = q.var(torch.ones(4, 3), requires_grad=True).v
    a = q.var(torch.ones(4, 3), requires_grad=False).v

    tfm = DummyTFModule2()
    y, _ = tfm(x, a)
    print(y)
    l = y.sum()
    print("doing backward")
    l.backward()
    print("x grad")
    print(x.grad)
    print("done")
    print(tfm._params[0])
    print(tfm._params[0].grad)
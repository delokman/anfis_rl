import torch
from graphviz import Digraph


# From https://discuss.pytorch.org/t/print-autograd-graph/692/15

def get_gradient_values(var):
    seen = set()

    parameter_values = dict()

    def add_gradient(var, t, name, i, o):
        # print("1", id(var), t, type(i), type(o), len(i), len(o))
        # i = i[0]

        ins = []

        for v in i:
            if v is not None:
                ins.append((v.min().item(), v.max().item()))
            else:
                ins.append(None)

        parameter_values[name] = {
            'in': ins,
            'out': [(o[0].min().item(), o[0].max().item())]
        }

    def add_g(var, t, name, i):
        # print("2", id(var), t, type(i))

        parameter_values[name] = {
            "in": [],
            'out': [(i.min().item(), i.max().item())]
        }

    def add_nodes(var, name):
        if var not in seen:
            if torch.is_tensor(var):
                var.register_hook(lambda i, o: add_gradient(var, "Tensor", name, i, o))
            elif hasattr(var, 'variable'):
                u = var.variable
                var.register_hook(lambda i, o: add_gradient(var, "Has variable", name, i, o))
                u.register_hook(lambda i: add_g(var, "Variable", name, i))
                # node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                # node_name = '%s\n min: %f max: %f \n %s' % (
                # param_map.get(id(u)), u.grad.min().item(), u.grad.max().item(), size_to_str(u.size()))
                # dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                var.register_hook(lambda i, o: add_gradient(var, "Generic", name, i, o))
                # dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for i, u in enumerate(var.next_functions):
                    if u[0] is not None:
                        # dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0], name + str(type(u[0]).__name__) + str(i))
            if hasattr(var, 'saved_tensors'):
                for i, t in enumerate(var.saved_tensors):
                    add_nodes(t, name + str(type(t).__name__) + str(i))

    add_nodes(var.grad_fn, str(type(var.grad_fn).__name__))
    return parameter_values


def make_dot(var, params, parameter_values=None):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Example implementation

    >>> model = CNN()
    >>> policy_loss = loss_fnc(y, y_pred)
    >>> g = make_dot(policy_loss, dict(model.named_parameters()))
    >>> g.view()

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    # print(param_map)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + ', '.join(['%d' % v for v in size]) + ')'

    def get_attr(id):
        if (parameter_values is None) or (id not in parameter_values):
            # print("Unkown ID", id)
            return ''

        attributes = []
        p = parameter_values[id]

        for key in ['in', 'out']:
            for i, x in enumerate(p[key]):
                if x is not None:
                    if isinstance(x, tuple):
                        attributes.append('%s %d: (%f, %f)' % (key, i, x[0], x[1]))
                        # attributes.append(('%s%d' % (key, i), "(%f, %f)" % (x[0], x[1])))
                    else:
                        # attributes.append(('%s%d' % (key, i), str(x)))
                        attributes.append('%s %d: %f' % (key, i, x))
                else:
                    # attributes.append(('%s%d' % (key, i), "None"))
                    attributes.append('%s %d: None' % (key, i))

        return '\n' + '\n'.join(attributes)

    def add_nodes(var, name):
        if var not in seen:
            if torch.is_tensor(var):
                node_name = size_to_str(var.size())

                attributes = get_attr(name)

                node_name += attributes

                dot.node(str(id(var)), node_name,
                         fillcolor='orange')

            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))

                attributes = get_attr(name)

                node_name += attributes

                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                node_name = "%s" % (
                    str(type(var).__name__))

                attributes = get_attr(name)

                node_name += attributes

                dot.node(str(id(var)), node_name)
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for i, u in enumerate(var.next_functions):
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0], name + str(type(u[0]).__name__) + str(i))
            if hasattr(var, 'saved_tensors'):
                for i, t in enumerate(var.saved_tensors):
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t, name + str(type(t).__name__) + str(i))

    add_nodes(var.grad_fn, str(type(var.grad_fn).__name__))
    return dot

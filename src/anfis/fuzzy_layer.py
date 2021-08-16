import torch


class JointFuzzifyLayer(torch.nn.Module):
    """
        A list of fuzzy variables, representing the inputs to the FIS.
        Forward pass will fuzzify each variable individually.
        We pad the variables so they all seem to have the same number of MFs,
        as this allows us to put all results in the same tensor.
    """

    def __init__(self, varmfs, varnames=None):
        super(JointFuzzifyLayer, self).__init__()
        if not varnames:
            self.varnames = ['x{}'.format(i) for i in range(len(varmfs))]
        else:
            self.varnames = list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))

    @property
    def num_in(self):
        """Return the number of input variables"""
        return len(self.varmfs)

    @property
    def max_mfs(self):
        """ Return the max number of MFs in any variable"""
        return max([var.num_mfs for var in self.varmfs.values()])

    def __repr__(self):
        """
            Print the variables, MFS and their parameters (for info only)
        """
        r = ['Input variables']
        for varname, members in self.varmfs.items():
            r.append('Variable {}'.format(varname))
            for mfname, mfdef in members.mfdefs.items():
                r.append('- {}: {}({})'.format(mfname,
                                               mfdef.__class__.__name__,
                                               ', '.join(['{}={}'.format(n, p.item())
                                                          for n, p in mfdef.named_parameters()])))
        return '\n'.join(r)

    def forward(self, x):
        """ Fuzzyify each variable's value using each of its corresponding mfs.
            x.shape = n_cases * n_in
            y.shape = n_cases * n_in * n_mfs
        """
        torch._assert(x.shape[1] == self.num_in, '{} is wrong no. of input values'.format(self.num_in))
        y_pred = torch.stack([var(x[:, i:i + 1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)
        return y_pred

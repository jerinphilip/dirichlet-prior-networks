
def in_ipynb():
    try:
        cls_name = get_ipython().__class__.__name__ 
        expected = 'ZMQInteractiveShell'
        flag = (cls_name == expected)
        return flag
    except NameError:
        return False

def flush_plot(plt, fname):
    plt.savefig(fname)
    if in_ipynb(): 
        plt.show()
    plt.clf()

if in_ipynb():
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

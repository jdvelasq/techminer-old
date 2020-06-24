def prueba():
    """
    >>> import os
    >>> print(os.getcwd())
    /workspaces/techminer
    >>> prueba()


    """
    n = 0
    for i in range(10):
        n += i
    print("hola")


if __name__ == "__main__":

    import doctest

    doctest.testmod()

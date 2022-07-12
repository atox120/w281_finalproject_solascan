## Util functions

def input_check(indict, key, default, out_dict, exception=False):
    """
    Checks that the required inputs are parsed to the output dictionary.
    If exception = True, then this is a required input and an error is raised. 
    Else default parameter is assigned to that parameter.
    """
    try:
        out_dict[key] = indict[key]
        del indict[key]
    except KeyError:
        if exception:
            raise KeyError(f'{key} is a required input and was not provided')
        else:
            if default is not None:
                out_dict[key] = default
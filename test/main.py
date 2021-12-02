if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import testcases


def main():
    testcases.backward_flow_2()

if __name__ == '__main__':
    main()

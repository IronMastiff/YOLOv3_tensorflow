import yaml

class Flag( object ):
    def __init__( self, ** entries ):
        self.__dict__.update( entries )

def read_config_file( config_file ):
    with open( config_file )as f:
        FLAG = Flag( **yaml.load( f ) )

    return FLAG
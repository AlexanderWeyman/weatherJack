#import json
import sys
from ibmpairs import paw, authentication


class ibmPAIRSWrapper(object):
    def __init__(self, server="https://pairs.res.ibm.com", api_key=None):
        # implement additional wrappers for ibm authorization methods here
        # i.e. not only api_key authorization
        self.__AUTH_TYPE = "api-key"
        if api_key == None:
            sys.stderr.writelines("Authorization incomplete. Exit.\n")
            sys.exit(1)
        self.__PAIRS_SERVER = server
        if api_key!="0":
            self.__PAIRS_CREDENTIALS = authentication.OAuth2(api_key=api_key)
        
        
    def query(self, query_string):
        return paw.PAIRSQuery(query_string, self.__PAIRS_SERVER, self.__PAIRS_CREDENTIALS, authType=self.__AUTH_TYPE)
        
        

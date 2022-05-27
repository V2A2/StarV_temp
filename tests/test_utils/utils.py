import sys
import traceback

def process_exception(ex): 
    ex_type, ex_value, ex_traceback = sys.exc_info()
    trace_back = traceback.extract_tb(ex_traceback)
        
    stack_trace = ""
        
    for trace in trace_back:
        stack_trace = stack_trace + "File : %s ,\n Line : %d,\n Func.Name : %s,\n Message : %s\n" % (trace[0], trace[1], trace[2], trace[3])
                
    print("Exception type : %s " % ex_type.__name__)
    print("Exception message : %s" %ex_value)
    print("Stack trace : %s" %stack_trace)

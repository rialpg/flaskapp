from flask import Flask,request,render_template

app=Flask(__name__)

class Alert:
    def __init__(self,SERVER_ID, CPU_UTILIZATION, MEMORY_UTILIZATION, DISK_UTILIZATION):
        self.SERVER_ID = SERVER_ID
        self.CPU_UTILIZATION =CPU_UTILIZATION
        self.MEMORY_UTILIZATION=MEMORY_UTILIZATION
        self.DISK_UTILIZATION=DISK_UTILIZATION
    def CheckRules(self):
        val=[0,0,0]
        if(self.CPU_UTILIZATION>85):
            val[0]=1
        if(self.MEMORY_UTILIZATION>75):
            val[1]=1
        if(self.DISK_UTILIZATION>60):
            val[2]=1
        return val


def Driver(SERVER_ID, CPU_UTILIZATION, MEMORY_UTILIZATION, DISK_UTILIZATION):
    obj=Alert(float(SERVER_ID), float(CPU_UTILIZATION), float(MEMORY_UTILIZATION), float(DISK_UTILIZATION))
    val=obj.CheckRules()
    res=''
    if(val[0] or val[1] or val[2]):
        res=res+"Alert,"
    else:
        res=res+"No Alert,"
    res+=str(SERVER_ID)
    if(val[0]):
        res+=",CPU UTILIZATION VIOLATED"
    if(val[1]):
        res+=",MEMORY UTILIZATION VIOLATED"
    if(val[2]):
        res+=",DISK UTILIZATION VIOLATED"
    return res

@app.route('/',methods=['POST','GET'])
def index():

    if request.method=='GET':
        return render_template('index.html')
    else:
        SERVER_ID=request.form['SERVER_ID']
        CPU_UTILIZATION=request.form['CPU_UTILIZATION']
        MEMORY_UTILIZATION=request.form['MEMORY_UTILIZATION']
        DISK_UTILIZATION=request.form['DISK_UTILIZATION']
        res=Driver(SERVER_ID, CPU_UTILIZATION, MEMORY_UTILIZATION, DISK_UTILIZATION)
        #val=[1,1,1]
        return '<h1>{}</h1>'.format(res)

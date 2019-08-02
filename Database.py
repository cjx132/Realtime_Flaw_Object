import sqlite3 as lite
from enum import Enum

from PyQt5.QtGui import QPixmap

class Database():
    __instance = None

    @staticmethod
    def getInstance():
        if Database.__instance is None:
            Database()
        return Database.__instance

    def __init__(self):
        if Database.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Database.__instance = self
            self.con = lite.connect("database/Mainbase.db")
    def getCaminfo(self,ID):
        if ID is not None:
            command = "select camid, coordinate_x, coordinate_y,rate,camip,camport,username,userpassword from camera where camid = '{}'".format(str(ID))
        else:
            command = "select camid, coordinate_x, coordinate_y,rate,camip,camport,username,userpassword from camera where camid = 'cam_01'"
        cur = self.con.cursor()
        cur.execute(command)
        rows = cur.fetchall()
        ret = [(row[4], row[5], row[6], row[7],row[0], row[3], row[1], row[2]) for row in rows]
        cur.close()
        return ret

    def getCamurl(self,ID):
        if ID is not None:
            command = "select camurl from camera where camid = '{}'".format(str(ID))
        else:
            command = "select camurl from camera where camid = 'cam_01'"
        cur = self.con.cursor()
        cur.execute(command)
        rows = cur.fetchall()
        ret = [(row[0]) for row in rows]
        cur.close()
        return ret

    def getOpenflag(self,ID):
        if ID is not None:
            command = "select isopen from camera where camid = '{}'".format(str(ID))
        else:
            command = "select isopen from camera where camid = 'cam_01'"
        cur = self.con.cursor()
        cur.execute(command)
        rows = cur.fetchall()
        ret = [(row[0]) for row in rows]
        cur.close()
        return ret

    def updataCaminfo(self,camid='cam_01',coordinate_x=0.0,coordinate_y=0.0,rate=1,camip='',camport='',username='',userpassword=''):
        sql= "update camera set coordinate_x=?,coordinate_y=?,rate=?,camurl=?,camip=?,camport=?,username=?,userpassword=? where camid='{}'".format(str(camid))
        cur = self.con.cursor()
        if camip=='':
            camurl=''
        else:
            camurl="rtsp://" + str(username) + ":" + str(userpassword) + "@" + str(camip) + ":" + str(
                    camport) + "/h264/ch33/main/av_stream"
        cur.execute(sql, (coordinate_x,coordinate_y,rate,camurl,camip,camport,username,userpassword))
        cur.close()
        self.con.commit() 

    def updataOpenflag(self,camid='cam_01',isopen = 0):
        sql= "update camera set isopen= ? where camid='{}' ".format(str(camid))
        cur = self.con.cursor()
        cur.execute(sql, [(isopen)])
        cur.close()
        self.con.commit() 

    def insertIntoCamera(self,camid='cam_01',coordinate_x=0.0,coordinate_y=0.0,rate=1,camip='',camport='',username='',userpassword='',isopen=0):
        sql = "INSERT INTO camera(camid,coordinate_x,coordinate_y,rate,camurl,camip,camport,username,userpassword,isopen) VALUES(?,?,?,?,?,?,?,?,?,?)"
        cur = self.con.cursor()
        camurl = ''
        cur.execute(sql, (camid,coordinate_x,coordinate_y,rate,camurl,camip,camport,username,userpassword,isopen))
        cur.close()
        self.con.commit()
    
    #对瑕疵表的操作
    def insertIntoFlaw(self,flaw_id=1000,flaw_type='',camera_id='',coordinate_x=0.0,coordinate_y=0.0,width=0.0,highth=0.0,flaw_time='0000-00-00 00:00:00',cloth_type=''):
        sql = "INSERT INTO Flaw(flaw_id,flaw_type,camera_id,coordinate_x,coordinate_y,width,highth,flaw_time,cloth_type) VALUES(?,?,?,?,?,?,?,?,?)"
        cur = self.con.cursor()
        cur.execute(sql,(flaw_id,flaw_type,camera_id,coordinate_x,coordinate_y,width,highth,flaw_time,cloth_type))
        cur.close()
        self.con.commit()
    
    def getFlawCount(self):
        command = 'SELECT flaw_id FROM Flaw'
        cur = self.con.cursor()
        cur.execute(command)
        rows = cur.fetchall()
        ret = [(row[0]) for row in rows]
        cur.close()
        return len(ret)
        
    def getDataFromFlaw(self,flaw_type=""):
        command = "select flaw_id,flaw_type,camera_id,coordinate_x,coordinate_y,width,highth,flaw_time from Flaw where flaw_type='{}'".format(str(flaw_type))
        cur = self.con.cursor()
        cur.execute(command)
        rows = cur.fetchall()
        ret = [(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7]) for row in rows]
        cur.close()
        return ret

    def insertIntoFlawStatistic(self,flaw_type='',flaw_cont=0):
        sql = "INSERT INTO FlawStatistic(flaw_type,flaw_cont) VALUES(?,?)"
        cur = self.con.cursor()
        cur.execute(sql,(flaw_type,flaw_cont))
        cur.close()
        self.con.commit()
    
    def updateFlawStatistic(self,flaw_type='',flaw_cont=0):
        sql = "UPDATE FlawStatistic SET flaw_cont= ? where flaw_type='{}'".format(str(flaw_type))
        cur = self.con.cursor()
        cur.execute(sql,[(flaw_cont)])
        cur.close()
        self.con.commit()
    
    def getcntFromFlawStatistic(self,flaw_type=""):
        command = "select flaw_cont from FlawStatistic where flaw_type = '{}'".format(str(flaw_type))
        cur = self.con.cursor()
        cur.execute(command)
        rows = cur.fetchall()
        ret = [(row[0]) for row in rows]
        cur.close()
        return ret[0]
    
    def getallFromFlawStatistic(self):
        command = "select flaw_type,flaw_cont from FlawStatistic"
        cur = self.con.cursor()
        cur.execute(command)
        rows = cur.fetchall()
        ret = [(row[0],row[1]) for row in rows]
        cur.close()
        return ret


    #对瑕疵统计表的操作
    # def insertIntoFlawStatistic(self,flaw_label=1,flaw_type=''):
    #     sql = "INSERT INTO FlawStatistic(flaw_label,flaw_type) VALUES(?,?)"
    #     cur = self.con.cursor()
    #     cur.execute(sql,(flaw_label,flaw_type))
    #     cur.close()
    #     self.con.commit()
    # def getFlawCount(self):
    #     command = 'SELECT flaw_type,count(flaw_label) AS flaw_count FROM FlawStatistic GROUP BY flaw_label'
    #     cur = self.con.cursor()
    #     coursor =cur.execute(command)
    #     cur.close()
    #     return coursor
    #     #使用方法
    #     #for row in coursor:
    #     #   row[0]  瑕疵类型
    #     #   row[1]  瑕疵数量
    # #对设置表的操作
    def insertIntoSetting(self,model='yolo',width=0.0,highth=0.0,rate=1.0):
        sql = "INSERT INTO Setting(model,width,highth,rate) VALUES(?,?,?,?)"
        cur = self.con.cursor()
        cur.execute(sql,(model,width,highth,rate))
        cur.close()
        self.con.commit()
    def updateSetting(self,model='yolo',width=0.0,higth=0.0,rate=1.0):
        sql = "UPDATE Setting SET width=?,highth=?,rate=? where model='{}'".format(model)
        cur = self.con.cursor()
        cur.execute(sql, (width,higth,rate))
        cur.close()
        self.con.commit()
    
    def getSetting(self,model='yolo'):
        command = "select width,highth,rate from Setting where model = '{}'".format(model)
        cur = self.con.cursor()
        cur.execute(command)
        rows = cur.fetchall()
        ret = [(row[0],row[1],row[2]) for row in rows]
        cur.close()
        return ret[0]

    
    def deleteAllFlaw(self):
        commad = "delete from Flaw"
        cur = self.con.cursor()
        cur.execute(commad)
        cur.close()
        self.con.commit()
    
    def deleteAllFlawStatistic(self):
        commad = "delete from FlawStatistic"
        cur = self.con.cursor()
        cur.execute(commad)
        cur.close()
        self.con.commit()

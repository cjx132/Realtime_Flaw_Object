create table camera
(
	camid varchar(10) default 'cam_01' not null
		primary key,
	coordinate_x real,
	coordinate_y real,
    rate real,
    camurl VARCHAR(255),
    camip VARCHAR(20),
    camport VARCHAR(10),
    username VARCHAR(20),
    userpassword VARCHAR(20),
    isopen bit default 0
)
;
create table Flaw
(
    flaw_id int default 1000 not null
        primary key,
    flaw_type varchar(20),
    camera_id varchar(10),
    coordinate_x real,
	coordinate_y real,
    width real,
    highth real,
    flaw_time TIMESTAMP,
    cloth_type varchar(20)
)
;
create table FlawStatistic
(
    flaw_type varchar(20) not null
        primary key,
    flaw_cont INT default 0
)
;
create table Setting
(
    model varchar(20) default 'yolo' not null
        primary key,
    width real,
    highth real,
    rate real
)
;
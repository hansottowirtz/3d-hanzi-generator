// from https://forum.openscad.org/Rods-between-3D-points-td13104.html

module line(p1,p2,r){ // draw ray between 2 specified points
  translate((p1+p2)/2)
    rotate([-acos((p2[2]-p1[2]) / norm(p1-p2)),0,
            -atan2(p2[0]-p1[0],p2[1]-p1[1])])
       cylinder(r1=r, r2=r, h=norm(p1-p2), center = true);
}

module cone(p1,p2,r1,r2){ // draw ray between 2 specified points
  translate((p1+p2)/2)
    rotate([-acos((p2[2]-p1[2]) / norm(p1-p2)),0,
            -atan2(p2[0]-p1[0],p2[1]-p1[1])])
       cylinder(r1=r1, r2=r2, h=norm(p1-p2), center = true);
}

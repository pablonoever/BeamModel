function [ outvec ] = BECAS_Becas2BEAM( filename, r, const,...
    props, ...
    utils )
%********************************************************
% File: BECAS_Becas2Hawc2.m
%   This function generates input to HAWC2 based on the cross section
%   analysis results from BECAS
%
% Syntax:
%   BECAS_Becas2Hawc2( filename, r, const, props )
%
% Input:
%   filename:  String holding the name of the file to which the HAWC2 data
%              is output. Set to false (boolean) if no file need to be printed.
%   r       :  Radial position of section
%   const   :  Structure with constitutive matrices (see
%              BECAS_Constitutive*)
%   props   :  Structure with constitutive matrices (see
%              BECAS_CrossSectionProps)
%   utils   :  Structure with all inputdata and other data necessary (see
%              BECAS_utils).
%   
% Output:   
%    HAWC cross sectional data vector, and optionally optionally
%    that vector to a file with filename
%           
% Calls:
%
% Date:
%   Version 1.0    07.02.2012   José Pedro Blasques
%
%   Version 2.0    07.09.2012   José Pedro Blasques: Removed BECAS_utils 
%   and changed the input to receive the utils structure. Changed the ouput
%   to pass the props structure.
%
% (c) DTU Wind Energy
%********************************************************

%%
fprintf(1,'> Started generating input for HAWC2 and writing to BECAS2HAWC2.out...')

%Mass per unit length
mpl= const.Ms(1,1);

%Mass center coordinates with respect to c/2
xm= props.MassX;
ym= props.MassY;

%Radius of inertia with respect to elastic center
p=[ props.ElasticX  props.ElasticY]; theta=rad2deg( props.AlphaPrincipleAxis_ElasticCenter);
[Msprime]=BECAS_TransformCrossSectionMatrix(const.Ms,p,theta);
rx=sqrt(Msprime(4,4)/ mpl);
ry=sqrt(Msprime(5,5)/ mpl);

%Shear center coordinates with respect to c/2
xs= props.ShearX;
ys= props.ShearY;

%Modulus of elasticity (averaged)
%p=[ props.ElasticX  props.ElasticY]; theta=rad2deg( props.AlphaPrincipleAxis_ElasticCenter);
%[Ksprime]=BECAS_TransformCrossSectionMatrix(const.Ks,p,theta);
Ksprime = const.Ks;
Em=Ksprime(3,3)/ props.AreaTotal;
%Shear modulus (averaged)
% Em=0;
Gm=0;
nQ=(6)*(6);
for e=1:utils.ne_2d
    Qe=sparse(utils.iQ((e-1)*nQ+1:e*nQ),utils.jQ((e-1)*nQ+1:e*nQ),utils.vQ((e-1)*nQ+1:e*nQ));
    Gm=Gm+utils.ElArea(e)*Qe(4,4);
end
Gm=Gm/props.AreaTotal;

%Area moment of inertia with respect to principal bending axis
Ax_ea=Ksprime(4,4)/Em;
Ay_ea=Ksprime(5,5)/Em;

Axy_num = sum(utils.ElCent(:,1).*utils.ElCent(:,2).*utils.ElArea);

%Torsional stiffness
%p=[ props.ShearX  props.ShearY]; theta=rad2deg(0);
%[Ksprime]=BECAS_TransformCrossSectionMatrix(const.Ks,p,theta);
K=Ksprime(6,6)/Gm;

%Shear factor
kx=Ksprime(1,1)/(Gm* props.AreaTotal);
ky=Ksprime(2,2)/(Gm* props.AreaTotal);

%Cross section area
A= props.AreaTotal;

%Structural pitch
theta_s=rad2deg(props.AlphaPrincipleAxis_ElasticCenter);

%Elastic center position
xe= props.ElasticX;
ye= props.ElasticY;

%Poison ratio
vm = Em/(2*Gm);
if vm > 1
    vm = vm - 1;
end

%Output to HAWC2
outvec = [r mpl A xm ym xs ys Em Gm vm Ax_ea Ay_ea Axy_num K];

%Print to file, but use robust way to construct the full output path
% Linux uses / to separate folders, Windows uses \
% only print to a file when filename is not false

% Get file information
if filename ~= false
    filename=fullfile(pwd, filename);
    fileInfo = dir(filename);
end

% Check if the file exists and is not empty
if isempty(fileInfo) || fileInfo.bytes == 0
    fid = fopen(filename,'a+');
    format = repmat('%19s,',1,14);
	fprintf(fid,[format(1:end-1),'\n'], 'R', 'm', 'A', 'CGx', 'CGy', 'SCx', 'SCY', 'E', 'G', 'v', 'Ixx', 'Iyy', 'Ixy', 'J');
    fclose(fid);
end


fid = fopen(filename,'a+');
format = repmat('%19.12f,',1,14);
fprintf(fid,[format(1:end-1),'\n'], outvec);
fclose(fid);

    function [var]=rad2deg(var)
        %Turning radians to degrees
        var=var*180/(pi);
    end

fprintf(1,'DONE! \n');

end

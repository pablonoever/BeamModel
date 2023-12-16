
inp_folder = {'\BECAS\BECASv3.0\examples\Square Abaqus 10x10'};
R = [10];

n = length(R);

out_folder = 'test.txt';

for i = 1:n
    options.foldername = inp_folder{i};
    options.etype = 'Q4';
    [ utils ] = BECAS_Utils(options);
    const.Ms = BECAS_Constitutive_Ms(utils);
    const.Ks = BECAS_Constitutive_Ks(utils);
    props = BECAS_CrossSectionProps(const.Ks,utils);
    BECAS_Becas2BEAM(out_folder, R(i), const, props, utils )
    BECAS_PlotFEmesh(utils);
end
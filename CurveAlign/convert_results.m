res_dir = './eliceiri_4levels/HE/';
tforms = dir([res_dir, 'tform_*.mat']);

for i=1:length(tforms)
    t_name = tforms(i).name;
    load([res_dir, t_name], 'tform');
    t = tform.T;
    save([res_dir, t_name], 't');
end
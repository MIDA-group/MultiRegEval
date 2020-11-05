function CurveAlign

% CurveAlign.m - CurveAlign is an open-source quantitative tool for interpreting 
% the regional interaction between collagen and tumors by assessment of up to 
% ~thirty fiber % features, including angle, alignment, and density.

% There are two fiber analysis modes: 1) directly extract individual fibers 
% using an improved fiber tracking algorithm(CT-FIRE) based on Curvelet transform(CT) 
% and fiber extraction algorithm(FIRE); 2) directly find optimal fiber edges 
% representation using curvelet transform coefficients.

% CurveAlign allows the user to measure fiber alignment on a global, region 
% of interest (ROI), and fiber basis. Additionally, users can measure fiber
% alignment relative to manually or automatically segmented boundaries. This 
% tool does not require prior experience of programing or image processing 
% and can handle multiple files, enabling efficient quantification of collagen 
% organization from biological datasets.

%By Laboratory for Optical and Computational Instrumentation, UW-Madison
%since 2009
%Developers:
% Yuming Liu (primary contact and lead developer, Aug 2014-)
% Adib Keikhosravi (current graduate student developer, Aug 2014-)
% Guneet Singh Mehta (current graduate student developer, Aug 2014-)
% Jeremy Bredfeldt (former LOCI PhD student, Jun 2012-  Jul 2014)
% Carolyn Pehlke (former LOCI PhD student, Jan 2009- May 2012)

% Webpage: http://loci.wisc.edu/software/curvealign
% github: https://github.com/uw-loci/curvelets

% References:
%1. Schneider, C.A., Pehlke, C.A., Tilbury, K., Sullivan, R., Eliceiri, K.W., 
%   and Keely, P.J. (2013). Quantitative Approaches for Studying the Role of 
%   Collagen in Breast Cancer Invasion and Progression. In Second Harmonic 
%   Generation Imaging, F.S. Pavone, and P.J. Campagnola, eds. (New York: CRC Press), p. 373.
%2. Bredfeldt, J.S., Liu, Y., Conklin, M.W., Keely, P.J., Mackie, T.R., and Eliceiri, K.W. (2014).
%   Automated quantification of aligned collagen for human breast carcinoma prognosis. J Pathol Inform 5.
%3.  Liu, Y., Keikhosravi, A., Mehta, G.S., Drifka, C.R., and Eliceiri, K.W. (accepted).
%   Methods for quantifying fibrillar collagen alignment. In Fibrosis: Methods and Protocols, L. Ritti?, ed. (New York: Springer)

% Licensed under the 2-Clause BSD license 
% Copyright (c) 2009 - 2017, Board of Regents of the University of Wisconsin-Madison
% All rights reserved.

clc; home; clear all; close all;

if ~isdeployed
    addpath('./CircStat2012a','../../CurveLab-2.1.2/fdct_wrapping_matlab');
    addpath('./ctFIRE','./20130227_xlwrite','./xlscol/')
    addpath(genpath(fullfile('./FIRE')));
    display('Please make sure you have downloaded the Curvelets library from http://curvelet.org')
    %add Matlab Java path
    javaaddpath('./20130227_xlwrite/poi_library/poi-3.8-20120326.jar');
    javaaddpath('./20130227_xlwrite/poi_library/poi-ooxml-3.8-20120326.jar');
    javaaddpath('./20130227_xlwrite/poi_library/poi-ooxml-schemas-3.8-20120326.jar');
    javaaddpath('./20130227_xlwrite/poi_library/xmlbeans-2.3.0.jar');
    javaaddpath('./20130227_xlwrite/poi_library/dom4j-1.6.1.jar');
    javaaddpath('./20130227_xlwrite/poi_library/stax-api-1.0.1.jar');
end

%Get the actual screen size in pixels
set(0,'units','pixels');
ssU = get(0,'screensize'); % screen size of the user's display
set(0,'DefaultFigureWindowStyle','normal');

%Use the parameters from the last run of curveAlign
if exist('currentP_CA.mat','file')
    lastParamsGlobal = load('currentP_CA.mat');
    pathNameGlobal = lastParamsGlobal.pathNameGlobal;
    keepValGlobal = lastParamsGlobal.keepValGlobal;
    distValGlobal = lastParamsGlobal.distValGlobal;
    if isequal(pathNameGlobal,0)
        pathNameGlobal = '';
    end
    if isempty(keepValGlobal)
        keepValGlobal = 0.001;
    end
    if isempty(distValGlobal)
        distValGlobal = 150;
    end
else
    %use default parameters
    pathNameGlobal = '';
    keepValGlobal = 0.001;
    distValGlobal = 150;
end

% define the font size used in the GUI
fz1 = 9 ; % font size for the title of a panel
fz2 = 10; % font size for the text in a panel
fz3 = 12; % font size for the button
fz4 = 14; % biggest fontsize size

% provide advanced options
advancedOPT = struct('exclude_fibers_inmaskFLAG',1, 'curvelets_group_radius',10,...
    'seleted_scale',1,'heatmap_STDfilter_size',28,'heatmap_SQUAREmaxfilter_size',12,...
    'heatmap_GAUSSIANdiscfilter_sigma',4, 'plotrgbFLAG',0,'folderROIman','\\image path\ROI_management\',...
    'folderROIana','\\image path\ROI_management\Cropped\','uniROIname','',...
    'cropROI',0,'specifyROIsize',[256 256],'minimum_nearest_fibers',2,'minimum_box_size',32,'fiber_midpointEST',1);  

% set the pointer shape in manual csv boundary creation: PointerShapeCData
PointerShapeData = NaN*ones(16,16);
PointerShapeData(1:15,1:15) = 2*ones(15,15);
PointerShapeData(2:14,2:14) = ones(13,13);
PointerShapeData(3:13,3:13) = NaN*ones(11,11);
PointerShapeData(6:10,6:10) = 2*ones(5,5);
PointerShapeData(7:9,7:9) = 1*ones(3,3);

% Define the figures used for GUI
guiCtrl = figure('Resize','on','Units','normalized','Position',[0.002 0.09 0.25 0.85],...
    'Visible','off','MenuBar','none','name','CurveAlign V4.0 Beta','NumberTitle','off',...
    'UserData',0,'Tag','CurveAlign Main GUI');
double_click=0;
guiFig_norPOS = [0.255 0.09 0.711*ssU(4)/ssU(3) 0.711]; % normalized guiFig position
guiFig_absPOS = [guiFig_norPOS(1)*ssU(3) guiFig_norPOS(2)*ssU(4) guiFig_norPOS(3)*ssU(3) guiFig_norPOS(4)*ssU(4)]; %absolute guiFig position
 % CA and CAroi figure
guiFig = figure('Resize','on','Units','pixels','Position',guiFig_absPOS,...
    'Visible','off','MenuBar','figure','name','CurveAlign Figure','NumberTitle','off','UserData',0,...
    'KeyPressFcn',@roi_mang_keypress_fn);
guiRank1 = figure('Resize','on','Units','normalized','Position',[0.265 0.25 0.30 0.60],'Visible','off','MenuBar','none','name','CA Features List','NumberTitle','off','UserData',0);
guiRank2 = figure('Resize','on','Units','normalized','Position',[0.575 0.53 0.30 0.42],'Visible','off','MenuBar','figure','name','Feature Normalized Difference (Pos-Neg)','NumberTitle','off','UserData',0);
guiRank3 = figure('Resize','on','Units','normalized','Position',[0.575 0.02 0.30 0.42],'Visible','off','MenuBar','figure','name','Feature Classification Importance','NumberTitle','off','UserData',0);
defaultBackground = get(0,'defaultUicontrolBackgroundColor');
set(guiCtrl,'Color',defaultBackground);
set(guiFig,'Color',defaultBackground);
set(guiRank1,'Color',defaultBackground);
set(guiRank2,'Color',defaultBackground);
set(guiRank3,'Color',defaultBackground);
set(guiCtrl,'Visible','on');
imgPanel = uipanel('Parent', guiFig,'Units','normalized','Position',[0 0 1 1]);
imgAx = axes('Parent',imgPanel,'Units','normalized','Position',[0 0 1 1]);
guiFig2 = figure('Resize','on','Color',defaultBackground','Units','normalized',...
    'Position',[0.255 0.09 0.474*ssU(4)/ssU(3)*2 0.474],'Visible','off',...
    'MenuBar','figure','Name','CA output images','Tag', 'CA output images','NumberTitle','off','UserData',0);      % enable the Menu bar for additional operations
guiFig3 = figure('Resize','on','Color',defaultBackground','Units','normalized',...
    'Position',[0.255+0.474*ssU(4)/ssU(3) 0.09 0.474*ssU(4)/ssU(3) 0.474],'Visible','off',...
    'MenuBar','figure','name','CA ROI output Image','NumberTitle','off','UserData',0);      % enable the Menu bar for additional operations
guiFig4 = figure('Resize','on','Color',defaultBackground','Units','normalized',...
    'Position',[0.258+0.474*ssU(4)/ssU(3)*2 0.308 0.30*ssU(4)/ssU(3) 0.35],...
    'Visible','off','MenuBar','Figure','Name','CA angle distribution','Tag', 'CA angle distribution','NumberTitle','off','UserData',0);      % enable the Menu bar for additional operations

%Label for fiber mode drop down
fibModeLabel = uicontrol('Parent',guiCtrl,'Style','text','String','- Fiber analysis method',...
    'HorizontalAlignment','left','FontSize',fz2,'Units','normalized','Position',[0.5 .88 .5 .1]);
%drop down box for fiber analysis mode selection (CT-FIRE requires input data from CT-FIRE program)
fibModeDrop = uicontrol('Parent',guiCtrl,'Style','popupmenu','Enable','on','String',{'CT','CT-FIRE Segments','CT-FIRE Fibers','CT-FIRE Endpoints'},...
    'Units','normalized','Position',[.0 .88 .5 .1],'Callback',{@fibModeCallback});
%Label for boundary mode drop down
bndryModeLabel = uicontrol('Parent',guiCtrl,'Style','text','String','- Boundary method',...
    'HorizontalAlignment','left','FontSize',fz2,'Units','normalized','Position',[0.5 .84 .5 .1]);
%boundary mode drop down box, allows user to select which type of boundary analysis to do
bndryModeDrop = uicontrol('Parent',guiCtrl,'Style','popupmenu','Enable','on','String',{'No Boundary','CSV Boundary','Tiff Boundary'},...
    'Units','normalized','Position',[.0 .84 .5 .1],'Callback',{@bndryModeCallback});
% button to select an image file
imgOpen = uicontrol('Parent',guiCtrl,'Style','pushbutton','String','Get Image(s)','FontSize',fz3,'Units','normalized','Position',[0.01 .84 .45 .05],'callback','ClickedCallback','Callback', {@getFile});
imgLabel = uicontrol('Parent',guiCtrl,'Style','listbox','String','None Selected','HorizontalAlignment','left','FontSize',fz1,'Units','normalized','Position',[0.01 .685 .46 .145],'Callback', {@imgLabel_Callback});
% panel to contain other options
optPanel = uipanel('Parent',guiCtrl,'Title','RUN Options','Units','normalized','Position',[0.470 .680 0.530 0.218]);

%% CA ROI analysis button: ROI analysis button for CT/no boundary 
CAroi_man_button = uicontrol('Parent',optPanel,'Style','pushbutton','String','ROI Manager',...
    'FontSize',fz2,'UserData',[],'Units','normalized','Position',[0.01 0.67 0.48 0.30],...
    'callback','ClickedCallback','Callback', {@CAroi_man_Callback});
CAroi_ana_button = uicontrol('Parent',optPanel,'Style','pushbutton','String','ROI Analysis',...
    'FontSize',fz2,'UserData',[],'Units','normalized','Position',[0.51 0.67 0.48 0.30],...
    'callback','ClickedCallback','Callback', {@CAroi_ana_Callback});

%% Boundary creation button: create cvs open boundary 
CTF_module = uicontrol('Parent',optPanel,'Style','pushbutton','String','CT-FIRE',...
    'FontSize',fz2,'UserData',[],'Units','normalized','Position',[0.01 0.36 0.48 0.30],...
    'callback','ClickedCallback','Callback', {@CTFIRE_Callback});

% Boundary creation button: create tif boundary 
BDmask = uicontrol('Parent',optPanel,'Style','pushbutton','String','BD Creation',...
    'FontSize',fz2,'UserData',[],'Units','normalized','Position',[0.51 0.36 0.48 0.30],...
    'callback','ClickedCallback','Callback', {@BDmask_Callback});

BDCchoice = [];BW_shape = [];

% Post-processing button: post-processing CA extracted features
CAFEApost = uicontrol('Parent',optPanel,'Style','pushbutton','String','Post-Processing',...
    'FontSize',fz2,'UserData',[],'Units','normalized','Position',[0.01 0.05 0.48 0.30],...
    'callback','ClickedCallback','Callback', {@CAFEApost_Callback});

% feature ranking button: process an output feature mat files
fRanking = uicontrol('Parent',optPanel,'Style','pushbutton','String','Feature Ranking',...
    'FontSize',fz2,'UserData',[],'Units','normalized','Position',[0.51 0.05 0.48 0.30],...
    'callback','ClickedCallback','Callback', {@featR});

%button to set advanced options
advOptions = uicontrol('Parent',guiCtrl,'Style','pushbutton','String','Advanced','FontSize',fz4,'Units','normalized','Position',[0 .0 .32 .05],'Callback',{@advOptions_callback});

% button to run measurement
imgRun = uicontrol('Parent',guiCtrl,'Style','pushbutton','String','Run','FontSize',fz4,'Units','normalized','Position',[0.34 .0 .32 .05]);

% button to reset gui
 imgReset = uicontrol('Parent',guiCtrl,'Style','pushbutton','String','Reset','FontSize',fz4,'Units','normalized','Position',[.68 .0 .32 .05],'callback','ClickedCallback','Callback',{@resetImg});

% panel to contain output checkboxes
guiPanel0 = uipanel('Parent',guiCtrl,'Title','Primary Parameters ','Units','normalized','Position',[0 .45 1 .125],'Fontsize',fz1);

% text box for taking in curvelet threshold "keep"
keepLab1 = uicontrol('Parent',guiPanel0,'Style','text','String','Enter fraction of coefs to keep [in decimal](default is .001):','FontSize',fz2,'Units','normalized','Position',[0 .50 .75 .45]);
% keepLab2 = uicontrol('Parent',guiPanel0,'Style','text','String',' (default is .001)','FontSize',fz3,'Units','normalized','Position',[0.25 .50 .50 .25]);
enterKeep = uicontrol('Parent',guiPanel0,'Style','edit','String',num2str(keepValGlobal),'BackgroundColor','w','Min',0,'Max',1,'UserData',[keepValGlobal],'FontSize',fz3,'Units','normalized','Position',[.75 .55 .25 .45],'Callback',{@get_textbox_data});

distLab = uicontrol('Parent',guiPanel0,'Style','text','String','Enter distance from boundary to evaluate, in pixels:','FontSize',fz2,'Units','normalized','Position',[0 0 .75 .45]);
enterDistThresh = uicontrol('Parent',guiPanel0,'Style','edit','String',num2str(distValGlobal),'BackgroundColor','w','Min',0,'Max',1,'UserData',[distValGlobal],'FontSize',fz3,'Units','normalized','Position',[.75 0 .25 .45],'Callback',{@get_textbox_data2});

% panel to contain output checkboxes
guiPanel = uipanel('Parent',guiCtrl,'Title','Output Options','Units','normalized','Position',[0 .30 1 .125],'Fontsize',fz1);

% checkbox to display the image reconstructed from the thresholded
% curvelets
makeRecon = uicontrol('Parent',guiPanel,'Style','checkbox','Enable','off','String','Reconstructed Image','Min',0,'Max',3,'Units','normalized','Position',[.075 .66 .8 .3],'Fontsize',fz2);

% checkbox to display a histogram
makeAngle = uicontrol('Parent',guiPanel,'Style','checkbox','Enable','off','String','Angle Values&Histogram','UserData','0','Min',0,'Max',3,'Units','normalized','Position',[.075 .33 .8 .3],'Fontsize',fz2);

% % % checkbox to output list of values
%  makeValues = uicontrol('Parent',guiPanel,'Style','checkbox','Enable','off','String','Values','UserData','0','Min',0,'Max',3,'Units','normalized','Position',[.075 .5 .8 .1]);
% checkbox to show curvelet boundary associations
makeAssoc = uicontrol('Parent',guiPanel,'Style','checkbox','Enable','off','String','Bdry Assoc','UserData','0','Min',0,'Max',3,'Units','normalized','Position',[.075 .0 .8 .3],'Fontsize',fz2);
% checkbox to create a feature output file
makeFeat = uicontrol('Parent',guiPanel,'Style','checkbox','Enable','off','String','Feature List','UserData','0','Min',0,'Max',3,'Units','normalized','Position',[.6 .66 .8 .3],'Fontsize',fz2);
% checkbox to create an overlay image
makeOver = uicontrol('Parent',guiPanel,'Style','checkbox','Enable','off','String','Overlay Output','UserData','0','Min',0,'Max',3,'Units','normalized','Position',[.6 .33 .8 .3],'Fontsize',fz2);
% checkbox to create a map image
makeMap = uicontrol('Parent',guiPanel,'Style','checkbox','Enable','off','String','Map Output','UserData','0','Min',0,'Max',3,'Units','normalized','Position',[.6 .0 .8 .3],'Fontsize',fz2);
% listbox containing names of active files
%listLab = uicontrol('Parent',guiCtrl,'Style','text','String','Selected Images: ','FontUnits','normalized','FontSize',.2,'HorizontalAlignment','left','Units','normalized','Position',[0 .6 1 .1]);
%imgList = uicontrol('Parent',guiCtrl,'Style','listbox','BackgroundColor','w','Max',1,'Min',0,'Units','normalized','Position',[0 .425 1 .25]);
% slider for scrolling through stacks
slideLab = uicontrol('Parent',guiCtrl,'Style','text','String','Stack image selected:','Enable','off','FontSize',fz1,'Units','normalized','Position',[0 .60 .75 .08]);
stackSlide = uicontrol('Parent',guiCtrl,'Style','slide','Units','normalized','position',[0 .58 1 .075],'min',1,'max',100,'val',1,'SliderStep', [.1 .2],'Enable','off','Callback',{@slider_chng_img});
infoLabel = uicontrol('Parent',guiCtrl,'Style','text','String',strcat('  For feature extraction, choose ',...
     ' fiber analysis method and/or boudnary method, then click "Get Image(s)" button',...
      sprintf('\n  For pre/post-processing, click button in RUN options panel.')),...
     'FontSize',fz3,'Units','normalized','Position',[0 .065 1.0 .215],'BackgroundColor','g');
% set font
set([guiPanel keepLab1 distLab infoLabel enterKeep enterDistThresh makeRecon makeAngle makeAssoc imgOpen advOptions imgRun imgReset slideLab],'FontName','FixedWidth')
set([keepLab1 distLab],'ForegroundColor',[.5 .5 .5])
% set([imgOpen fRanking imgRun imgReset],'FontWeight','bold')
set([imgOpen advOptions imgRun imgReset],'FontWeight','bold')
set([keepLab1 distLab slideLab infoLabel],'HorizontalAlignment','left')
%initialize gui
set([imgRun makeAngle makeRecon enterKeep enterDistThresh advOptions],'Enable','off')
set([CAroi_man_button CAroi_ana_button],'Enable','off');
set([makeRecon makeAngle makeFeat makeOver makeMap],'Value',3)

% initialize variables used in some callback functions
index_selected = 1;   % default file index
idx = 1;             % index to the slice of a stack 
altkey = 0;   % 1: altkey is pressed
coords = [-1000 -1000];
aa = 1;
imgName = '';
imgSize = [0 0];
rows = [];
cols = [];
ff = '';
pathName = '';
fileName = '';
bndryFnd = '';
ctfFnd = '';
numSections = 0;
info = []; 
fileEXT = '.tif'; % default image format
%global flags, indicating the method chosen by the user
fibMode = 0;
bndryMode = 0;
bdryImg = [];
%text for the info box to help guide the user.
note1 = 'Click Get Image(s). ';
note2 = 'CT-FIRE file(s) must be in the sub-folder "\\image foder\ctFIREout\". ';
note3T = 'Tiff ';
note3C = 'CSV ';
note3 = 'boundary files must be in the sub-folder "\\image foder\CA_Boudary\" and conform to naming convention.';
% uicontrols for automatical boundary creation from RGB HE image
HEpathname = ''; 
HEfilename = '';
pixelpermicron = 5.0; 
areaThreshold = 5000;
SHGpathname = '';
BDCparameters = struct('HEfilepath',HEpathname,'HEfilename',HEfilename,'pixelpermicron',1.5,'areaThreshold',500,'SHGfilepath',SHGpathname);

BDCgcf = figure('Resize','on','Units','normalized','Position',[0.1 0.60 0.20 0.30],...
    'Visible','off','MenuBar','none','name','Automatic Boundary Creation',...
    'NumberTitle','off','UserData',0,'Tag','Boundary Creation');
% button to open HE files 
HEfileopen = uicontrol('Parent',BDCgcf,'Style','Pushbutton','String','Get HE Files','FontSize',fz1,'Units','normalized','Position',[0 .75 0.25 .15],'Callback',{@getHEfiles_Callback},'TooltipString',...
    'Click to select HE image files to be registered or segmented');
HEfileinfo = uicontrol('Parent',BDCgcf,'Style','edit','String','No file is selected.','FontSize',fz1,'Units','normalized',...
    'Position',[0.265 0.75 .725 .15],'Callback',{@enterHEfolder_Callback},'TooltipString',...
    'HE images should registered with SHG images and have the same file name as SHG');
% button to get SHG file folder with which the HE is registrated. 
SHGfolderopen = uicontrol('Parent',BDCgcf,'Style','Pushbutton','String','Get SHG Folder','FontSize',fz1,'Units','normalized','Position',[0 .58 0.25 .15],'Callback',{@getSHGfolder_Callback},'TooltipString',...
    'Click to select SHG folder with which the HE files are associated');
SHGfolderinfo = uicontrol('Parent',BDCgcf,'Style','edit','String','No folder is specified.','FontSize',fz1,'Units','normalized',...
    'Position',[0.265 0.58 .725 .15],'Callback',{@enterSHGfolder_Callback},'TooltipString',...
    'The HE files are registered with SHG files in this folder,Boundary mask folder will be in this directory.');
% edit box to update HE image resolution in pixel per micron
HE_RES_text = uicontrol('Parent',BDCgcf,'Style','text','String','Pixel/Micron','FontSize',fz1,'Units','normalized','Position',[0 .36 0.25 .15]);
HE_RES_edit = uicontrol('Parent',BDCgcf,'Style','edit','String',num2str(pixelpermicron),'FontSize',fz1,'Units','normalized','Position',[0.265 0.41 .225 .15],'Callback',{@HE_RES_edit_Callback});
% edit box to update area Threshold in pixel per micron
HE_threshold_text = uicontrol('Parent',BDCgcf,'Style','text','String',sprintf('Area Threshold\n(pixel^2)'),'FontSize',fz1,'Units','normalized','Position',[0 .19 0.25 .15]);
HE_threshold_edit = uicontrol('Parent',BDCgcf,'Style','edit','String',num2str(areaThreshold),'FontSize',fz1,'Units','normalized','Position',[0.265 0.24 .225 .15],'Callback',{@HE_threshold_edit_Callback});
set(HE_threshold_edit, 'Enable','off');
% checkbox to disply mask when a single HE image is loaded
HEmask_figureFLAG = uicontrol('Parent',BDCgcf,'Style','checkbox','Enable','off',...
    'String','Display','Value',3,'UserData',3,'Min',0,'Max',3,'Units','normalized',...
    'Position',[0.025 0.05 .20 .15],'Fontsize',fz1,'TooltipString','Display results');
% checkbox to registration when HE image needs to be registered
HEreg_FLAG = uicontrol('Parent',BDCgcf,'Style','checkbox','Enable','on','String','Reg',...
    'Value',0,'UserData',0,'Min',0,'Max',3,'Units','normalized','Position',[0.225 0.05 .20 .15],...
    'Fontsize',fz1,'TooltipString','Register HE bright field image with SHG image',...
    'Callback',{@HEreg_FLAG_Callback});
% checkbox to segment registered HE bright field image
HEseg_FLAG = uicontrol('Parent',BDCgcf,'Style','checkbox','Enable','on','String','Seg',...
    'Value',3,'UserData',3,'Min',0,'Max',3,'Units','normalized','Position',[0.425 0.05 .20 .15],...
    'Fontsize',fz1,'TooltipString','Segment registered HE bright field to create tumor boundary','Callback',{@HEseg_FLAG_Callback});
% BDCgcf ok  and cancel buttons 
BDCgcfOK = uicontrol('Parent',BDCgcf,'Style','Pushbutton','String','OK','FontSize',fz1,'Units','normalized','Position',[0.655 .05 0.15 .1],'Callback',{@BDCgcfOK_Callback});
BDCgcfCANCEL = uicontrol('Parent',BDCgcf,'Style','Pushbutton','String','Cancel','FontSize',fz1,'Units','normalized','Position',[0.815 .05 0.15 .1],'Callback',{@BDCgcfCANCEL_Callback});
set([HEfileinfo SHGfolderinfo],'BackgroundColor','w','Min',0,'Max',1,'HorizontalAlignment','left')
set([HE_RES_edit HE_threshold_edit],'BackgroundColor','w','Min',0,'Max',1,'HorizontalAlignment','center')
% CA post-processing gui
% uicontrols for automatical boundary creation from RGB HE image
CApostfolder = ''; 
CApostOptions = struct('CApostfilepath',CApostfolder,'RawdataFLAG',0,'ALLstatsFLAG',1,'SELstatsFLAG',0);
CApostgcf = figure('Resize','on','Units','normalized','Position',[0.1 0.50 0.20 0.40],...
    'Visible','off','MenuBar','none','name','Post-processing CA features','NumberTitle','off','UserData',0);
% button to open CA output folder 
CApostfolderopen = uicontrol('Parent',CApostgcf,'Style','Pushbutton','String','Get CA output folder','FontSize',fz1,'Units','normalized','Position',[0 .885 0.35 .075],'Callback',{@CApostfolderopen_Callback});
CApostfolderinfo = uicontrol('Parent',CApostgcf,'Style','text','String','No folder is selected.','FontSize',fz1,'Units','normalized','Position',[0.01 0.78 .98 .10]);
% panel to contain checkboxes of output options
guiPanel_CApost = uipanel('Parent',CApostgcf,'Title','Post-processing Options ','Units','normalized','FontSize',fz2,'Position',[0 0.25 1 .45]);
% statistics of all features
makeCAstats_all = uicontrol('Parent',guiPanel_CApost,'Style','checkbox','Enable','on','String','Mean of all features','Min',0,'Max',3,'Value',3,'Units','normalized','Position',[.075 .66 .8 .33],'FontSize',fz1);
% combine raw feature files
combine_featurefiles = uicontrol('Parent',guiPanel_CApost,'Style','checkbox','Enable','on','String','Combine feature files','Min',0,'Max',3,'Value',0,'Units','normalized','Position',[.075 .33 .8 .33],'FontSize',fz1);
% statistics of selected features
makeCAstats_exist = uicontrol('Parent',guiPanel_CApost,'Style','checkbox','Enable','on','String','Mean of selected features','Min',0,'Max',3,'Value',0,'Units','normalized','Position',[.075 0.01 .8 .33],'FontSize',fz1);
% BDCgcf ok  and cancel buttons 
CApostgcfOK = uicontrol('Parent',CApostgcf,'Style','Pushbutton','String','OK','FontSize',fz1,'Units','normalized','Position',[0.625 .05 0.15 .1],'Callback',{@CApostgcfOK_Callback});
CApostgcfCANCEL = uicontrol('Parent',CApostgcf,'Style','Pushbutton','String','Cancel','FontSize',fz1,'Units','normalized','Position',[0.815 .05 0.15 .1],'Callback',{@CApostgcfCANCEL_Callback});
set([CApostfolderinfo],'BackgroundColor','w','Min',0,'Max',1,'HorizontalAlignment','left')
% end post-processing GUI
img = [];  % current image data
% ROI analysis
%YL: define all the output files, directory here
ROIanaBatOutDir = '';
ROIimgDir = '';% 
ROImanDir = '';% 
ROIanaDir = '';% 
ROIDir = '';% 
ROIpostBatDir = '';% 
BoundaryDir = '';% 
roiMATnamefull = ''; % name of a ROI .mat file
roiMATnameV = ''; % name vector of all the ROI .mat files
loadROIFLAG = 0;  % 1: load ROI file from specified folder other than the default folder
ROIshapes = {'Rectangle','Freehand','Ellipse','Polygon'};
cropIMGon = 1;   % 1: use cropped ROI, 0: use ROI mask
%YL: add CA ROI analysis output table
    % Column names and column format
columnname = {'No.','Image Label','ROI label','Orentation','Alignment','FeatNum','Methods','Boundary','CROP','POST','Shape','Xc','Yc','Z'};
columnformat = {'numeric','char','char','char','char' ,'char','char','char','char','char','char','numeric','numeric','numeric'};
columnwidth = {30 100 60 70 70 60 60 60 40 40 60 30 30 30 };   %
selectedROWs = [];
CA_data_current = [];
     % Create the uitable
     figPOS = [0.255 0.04+0.474+0.135 0.474*ssU(4)/ssU(3)*2 0.85-0.474-0.135];
     CA_table_fig = figure('Units','normalized','Position',figPOS,'Visible','off',...
         'NumberTitle','off','name','CurveAlign output table');
     CA_output_table = uitable('Parent',CA_table_fig,'Units','normalized','Position',[0.05 0.05 0.9 0.9],...
    'Data', CA_data_current,...
    'ColumnName', columnname,...
    'ColumnFormat', columnformat,...
    'ColumnWidth',columnwidth,...
    'ColumnEditable', [false false false false false false false false false false false false false false],...
    'RowName',[],...
    'CellSelectionCallback',{@CAot_CellSelectionCallback});
%% Add histogram when check the output table
 CA_table_fig2 = figure('Units','normalized','Position',figPOS,'Visible','off',...
         'NumberTitle','off','name','CurveAlign output table');
%-------------------------------------------------------------------------
%output table callback functions
    function CAot_CellSelectionCallback(hobject, eventdata,handles)
        handles.currentCell=eventdata.Indices;
        selectedROWs = unique(handles.currentCell(:,1));
        if isempty(selectedROWs)
            disp('No image is selected in the output table.')
            return
        end
        selectedZ = CA_data_current(selectedROWs,14);
        for j = 1:length(selectedZ)
            Zv(j) = selectedZ{j};
        end
        if size(unique(Zv)) == 1
            zc = unique(Zv);
        else
            error('only display ROIs in the same section of a stack');   % also not support different images
        end
        if length(selectedROWs) > 1
            IMGnameV = CA_data_current(selectedROWs,2);
            uniqueName = strncmpi(IMGnameV{1},IMGnameV,length(IMGnameV{1}));
            if length(find(uniqueName == 0)) >=1
                error('only display ROIs in the same section of a stack or in the same image');
            else
                IMGname = IMGnameV{1};
            end
        elseif length(selectedROWs) == 1
            IMGname = CA_data_current{selectedROWs,2};
        end
        %% Close the last overlay and heatmap
        CA_OLfig_h = findobj(0,'Name','CurveAlign Fiber Overlay');
        CA_MAPfig_h = findobj(0,'Name','CurveAlign Angle Map');
        if ~isempty(CA_OLfig_h)
            close(CA_OLfig_h)
            disp('The last CurveAlign overlay figure is closed, displaying the selected overlay-heatmap pair')
        end
        if ~isempty(CA_MAPfig_h)
            close(CA_MAPfig_h)
            disp('The last CurveAlign Angle heatmap is closed, displaying the selected overlay-heatmap pair')
        end
%%
        if ~isempty(CA_data_current{selectedROWs(1),3})   % ROI analysis, ROI label is not empty
            roiMATnamefull = [IMGname,'_ROIs.mat'];
            load(fullfile(ROImanDir,roiMATnamefull),'separate_rois')
            ROInames = fieldnames(separate_rois);
            
            IMGnamefull = fullfile(pathName,[IMGname,fileEXT]);
            IMGinfo = imfinfo(IMGnamefull);
            numSections = numel(IMGinfo); % number of sections
            if numSections == 1
                img2 = imread(IMGnamefull);
            elseif numSections > 1
                img2 = imread(IMGnamefull,zc);
            end
            if size(img2,3) > 1
                img2 = rgb2gray(img2);
            end
            IMGO(:,:,1) = uint8(img2);
            IMGO(:,:,2) = uint8(img2);
            IMGO(:,:,3) = uint8(img2);
            cropFLAG_selected = unique(CA_data_current(selectedROWs,9));
            if size(cropFLAG_selected,1)~=1
                disp('Please select ROIs processed with the same method.')
                return
            elseif size(cropFLAG_selected,1)==1
                cropFLAG = cropFLAG_selected;
            end
            postFLAG_selected = unique(CA_data_current(selectedROWs,10));
            if size(postFLAG_selected,1)~=1
                disp('Please select ROIs processed with the same method.')
                return
            elseif size(postFLAG_selected,1)==1
                postFLAG = postFLAG_selected;
            end
            
            bndFLAG_selected = unique(CA_data_current(selectedROWs,8));
            if size(bndFLAG_selected,1)~=1
                disp('Please select ROIs processed with the same method.')
                return
            elseif size(bndFLAG_selected,1)==1
                bndFLAG = bndFLAG_selected;
            end
            
             %histogram
              guiFig4 = findobj(0,'Tag','CA angle distribution');
            if isempty(guiFig4)
                guiFig4 = figure('Resize','on','Color',defaultBackground','Units','normalized',...
                    'Position',[0.258+0.474*ssU(4)/ssU(3)*2 0.308 0.30*ssU(4)/ssU(3) 0.35],...
                    'Visible','off','MenuBar','figure','Name','CA angle distribution','Tag', 'CA angle distribution',...
                    'NumberTitle','off','UserData',0); 
            end
            
            if strcmp(cropFLAG,'YES')
                for i= 1:length(selectedROWs)
                    CAroi_name_selected =  CA_data_current(selectedROWs(i),3);
                    ROInameV{i} = CAroi_name_selected{1};
                    ROInumV{i} = cell2mat(CA_data_current(selectedROWs(i),1));
                    if numSections > 1
                        roiNamefullNE = [IMGname,sprintf('_s%d_',zc),CAroi_name_selected{1}];
                    elseif numSections == 1
                        roiNamefullNE = [IMGname,'_', CAroi_name_selected{1}];
                    end
                    % check the output values
                    csvName_ROI = fullfile(ROIanaBatOutDir,[roiNamefullNE '_values.csv']);
                    if exist(csvName_ROI,'file')
                        csv_temp = importdata(csvName_ROI);
                        csvdata_ROI{i} = csv_temp(:,1);
                    else
                        csvdata_ROI{i} = nan;
                    end
                    
                    olName = fullfile(ROIanaBatOutDir,[roiNamefullNE '_overlay.tiff']);
                    if exist(olName,'file')
                        IMGol = imread(olName);
                    else
                        data2=separate_rois.(CAroi_name_selected{1}).roi;
                        a=round(data2(1));b=round(data2(2));c=round(data2(3));d=round(data2(4));
                        ROIrecWidth = c; ROIrecHeight = d;
                        IMGol = zeros(ROIrecHeight,ROIrecWidth,3);
                    end
                    if separate_rois.(CAroi_name_selected{1}).shape == 1
                        data2=separate_rois.(CAroi_name_selected{1}).roi;
                        a=round(data2(1));b=round(data2(2));c=round(data2(3));d=round(data2(4));
                        IMGO(b:b+d-1,a:a+c-1,1) = IMGol(:,:,1);
                        IMGO(b:b+d-1,a:a+c-1,2) = IMGol(:,:,2);
                        IMGO(b:b+d-1,a:a+c-1,3) = IMGol(:,:,3);
                        xx(i) = a+c/2;  yy(i)= b+d/2; ROIind(i) = selectedROWs(i);
                        aa2(i) = a; bb(i) = b;cc(i) = c; dd(i) = d;
                    else
                        error('Cropped image ROI analysis for shapes other than rectangle is not availabe so far.');
                    end
                end
                guiFig3 = findobj(0,'Name', 'CA ROI output Image');
                if isempty(guiFig3)
                    guiFig3 = figure('Resize','on','Color',defaultBackground','Units','normalized','Position',...
                        [0.255+0.474*ssU(4)/ssU(3) 0.09 0.474*ssU(4)/ssU(3)*1 0.474],'Visible','off',...
                        'MenuBar','figure','Name','CA ROI output Image','NumberTitle','off','UserData',0);      % enable the Menu bar for additional operations
                end
                figure(guiFig3);
                imshow(IMGO);set(guiFig,'Name',IMGname); hold on;
                for i= 1:length(selectedROWs)
                    CAroi_name_selected =  CA_data_current(selectedROWs(i),3);
                    if separate_rois.(CAroi_name_selected{1}).shape == 1
                        rectangle('Position',[aa2(i) bb(i) cc(i) dd(i)],'EdgeColor','y','linewidth',3)
                    end
                    text(xx(i),yy(i),sprintf('%d',ROIind(i)),'fontsize', 10,'color','m')
                end
                hold off
               set(guiFig3, 'Units','normalized','Position',[0.255+0.474*ssU(4)/ssU(3) 0.09 0.474*ssU(4)/ssU(3)*1 0.474]);
                
            end
            if strcmp(cropFLAG,'NO')
                ii = 0; boundaryV = {};yy = []; xx = []; RV = [];
                for i= 1:length(selectedROWs)
                    CAroi_name_selected =  CA_data_current(selectedROWs(i),3);
                    ROInameV{i} = CAroi_name_selected{1};
                    ROInumV{i} = cell2mat(CA_data_current(selectedROWs(i),1));
                    if ~iscell(separate_rois.(CAroi_name_selected{1}).shape)
                        ii = ii + 1;
                        %                         CAroi_name_selected =  CA_data_current(selectedROWs(i),3);
                        if numSections > 1
                            roiNamefullNE = [IMGname,sprintf('_s%d_',zc),CAroi_name_selected{1}];
                        elseif numSections == 1
                            roiNamefullNE = [IMGname,'_', CAroi_name_selected{1}];
                        end
                        IMGol = [];
                        if strcmp(postFLAG,'NO')
                            csvName_ROI = fullfile(ROIanaBatOutDir,[roiNamefullNE '_values.csv']);
                            if exist(csvName_ROI,'file')
                                csv_temp = importdata(csvName_ROI);
                                csvdata_ROI{i} = csv_temp(:,1);
                            else
                                csvdata_ROI{i} = nan;
                            end

                            olName = fullfile(ROIanaBatOutDir,[roiNamefullNE '_overlay.tiff']);
                            if exist(olName,'file')
                                IMGol = imread(olName);
                            end
                        else
                            csvName_ROI = fullfile(ROIpostBatDir,[roiNamefullNE '_fibFeatures.csv']);
                            if exist(csvName_ROI,'file')
                                csv_temp = importdata(csvName_ROI);
                                if strcmp(bndFLAG,'NO')
                                   csvdata_ROI{i} = csv_temp(:,4);  % 4: absolute angle
                                elseif strcmp(bndFLAG,'YES')
                                   csvdata_ROI{i} = csv_temp(:,30);  % 4: nearest relative angle
                                end
                            else
                                csvdata_ROI{i} = nan;
                            end
                            olName = fullfile(pathName,'CA_Out',[IMGname '_overlay.tiff']);
                            if exist(olName,'file')
                                if numSections == 1
                                    IMGol = imread(olName);
                                elseif numSections > 1
                                    IMGol = imread(olName,zc);
                                end
                            end
                        end
                        if isempty(IMGol)
                            IMGol = zeros(size(IMGO));
                        end
                       
                        % replace the region of interest with the data in the
                        % ROI analysis output
                        boundary = separate_rois.(CAroi_name_selected{1}).boundary{1};
                        [x_min,y_min,x_max,y_max] = enclosing_rect_fn(fliplr(boundary));
                        a = x_min;  % x of upper left corner of the enclosing rectangle
                        b = y_min;   % y of upper left corner of the enclosing rectangle
                        c = x_max-x_min;  % width of the enclosing rectangle
                        d = y_max - y_min;  % height of the enclosing rectangle
                        IMGO(b:b+d-1,a:a+c-1,1) = IMGol(b:b+d-1,a:a+c-1,1);
                        IMGO(b:b+d-1,a:a+c-1,2) = IMGol(b:b+d-1,a:a+c-1,2);
                        IMGO(b:b+d-1,a:a+c-1,3) = IMGol(b:b+d-1,a:a+c-1,3);
                        boundaryV{ii} = boundary;
                        yy(ii) = separate_rois.(CAroi_name_selected{1}).xm;
                        xx(ii) = separate_rois.(CAroi_name_selected{1}).ym;
                        RV(ii) = i;
                        ROIind(ii) = selectedROWs(i);
                    else
                        disp('Selected ROI is a combined one and is not displayed.')
                    end
                end
                guiFig3 = findobj(0,'Name', 'CA ROI output Image');
                if isempty(guiFig3)
                    guiFig3 = figure('Resize','on','Color',defaultBackground','Units','normalized',...
                        'Position',[0.255+0.474*ssU(4)/ssU(3) 0.09 0.474*ssU(4)/ssU(3)*1 0.474],'Visible','off',...
                        'MenuBar','figure','Name','CA ROI output Image','NumberTitle','off','UserData',0); 
                end
                figure(guiFig3);
                imshow(IMGO); hold on;
                if ii > 0
                    for ii = 1:length(selectedROWs)
                        text(xx(ii),yy(ii),sprintf('%d',ROIind(ii)),'fontsize', 10,'color','m')
                        boundary = boundaryV{ii};
                        plot(boundary(:,2), boundary(:,1), 'm', 'LineWidth', 2);%boundary need not be dilated now because we are using plot function now
                        text(xx(ii),yy(ii),sprintf('%d',selectedROWs(RV(ii))),'fontsize', 10,'color','m')
                    end
                    axis off
                else
                    disp('NO ROI analysis output is visulized')
                end
                 set(guiFig3,'Units','normalized','Position',[0.255+0.474*ssU(4)/ssU(3) 0.09 0.474*ssU(4)/ssU(3)*1 0.474]);  % update the postion for possible change after imshow
            end
            % histogram
            figure(guiFig4)
            htabgroup = uitabgroup(guiFig4);
            for i = 1: length(selectedROWs)
                tabfig_name1 = uitab(htabgroup, 'Title',...
                    sprintf('%d-Hist',ROInumV{i}));
                hax_hist = axes('Parent', tabfig_name1);
                set(hax_hist,'Position',[0.15 0.15 0.80 0.80]);
                output_values = csvdata_ROI{i};
                if strcmp(CA_data_current{selectedROWs(1),8}, 'YES')  % with boundary
                    bins = 2.5:5:87.5;
                    hist(output_values,bins);
                    [n_ang xout_ang] = hist(output_values,bins);
                    xlim([0 90]);
                elseif strcmp(CA_data_current{selectedROWs(1),8}, 'NO')  % no boundary
                    bins = 2.5:5:177.5;
                    hist(output_values,bins);
                    [n_ang xout_ang] = hist(output_values,bins);
                    xlim([0 180]);
                end
                xlabel('Angle [degree]')
                ylabel('Frequency [#]')
                axis square
                
                %Compass plot
                U = cosd(xout_ang).*n_ang;
                V = sind(xout_ang).*n_ang;
                tabfig_name2 = uitab(htabgroup, 'Title',...
                    sprintf('%d-Compass',ROInumV{i}));
                hax_cmpp = axes('Parent', tabfig_name2);
                set(hax_cmpp,'Position',[0.1 0.10 0.80 0.80]);
                compass(U,V)
            end
            
        else     % full image analysis, ROI label is empty
            %% 
            IMGnamefull = fullfile(pathName,[IMGname,fileEXT]);
            IMGinfo = imfinfo(IMGnamefull);
            SZ = selectedZ{1};
            OLnamefull = fullfile(pathName, 'CA_Out',[IMGname,'_overlay.tiff']);
            MAPnamefull = fullfile(pathName, 'CA_Out',[IMGname,'_procmap.tiff']);
            OLinfo = imfinfo(OLnamefull);
            MAPinfo = imfinfo(MAPnamefull);
            if numel(IMGinfo)== 1
                Output_values_name = fullfile(pathName,'CA_Out',[IMGname,'_values.csv']);
            elseif numel(IMGinfo)> 1
                Output_values_name = fullfile(pathName,'CA_Out',[IMGname,'_s' num2str(SZ) '_values.csv']);
            end
            guiFig2 = findobj(0,'Tag','CA output images');
            if isempty(guiFig2)
                guiFig2 = figure('Resize','on','Color',defaultBackground','Units','normalized',...
                    'Position',[0.255 0.09 0.474*ssU(4)/ssU(3)*2 0.474],'Visible','off',...
                    'MenuBar','figure','Name','CA output images','Tag', 'CA output images',...
                    'NumberTitle','off','UserData',0);      % enable the Menu bar for additional operations
            end
          
            if numel(IMGinfo) == 1
                figure(guiFig2);
                set(guiFig2,'Name',['CA output images for ',IMGname,fileEXT]);
                % link the axes of the original and OL images to simply a visual inspection of the fiber extraction
                axLINK1(1)= subplot(1,2,1); set(axLINK1(1),'Position', [0.01 0.01 0.485 0.94]);
                imshow(OLnamefull,'border','tight');
                title(sprintf('Overlaid, %dx%d, RGB, %3.2fM',OLinfo.Width,...
                    OLinfo.Height,OLinfo.FileSize/10^6),'fontweight','normal','FontSize',10)
                axis image
                axLINK1(2)= subplot(1,2,2); set(axLINK1(2),'Position', [0.505 0.01 0.485 0.94]);
                imshow(MAPnamefull,'border','tight');
                title(sprintf('Heatmap of angle, %dx%d, %d-bit ,%3.2fM',MAPinfo.Width,...
                    MAPinfo.Height,MAPinfo.BitDepth,MAPinfo.FileSize/10^6),'fontweight','normal','FontSize',10)
                axis image
                linkaxes(axLINK1,'xy')
            elseif numel(IMGinfo) > 1
                figure(guiFig2);
                set(guiFig2,'Name',['CA output images for ',IMGname,fileEXT,', ',num2str(SZ),'/',num2str(numel(IMGinfo))]);
                % link the axes of the original and OL images to simply a visual inspection of the fiber extraction
                axLINK1(1)= subplot(1,2,1); set(axLINK1(1),'Position', [0.01 0.01 0.485 0.94]);
                OLdata = imread(OLnamefull,SZ);
                imshow(OLdata,'border','tight');
                title(sprintf('Overlaid, %dx%d, RGB, %3.2fM',OLinfo(SZ).Width,...
                    OLinfo(SZ).Height,OLinfo(SZ).FileSize/10^6),'fontweight','normal','FontSize',10)
                axis image
                axLINK1(2)= subplot(1,2,2); set(axLINK1(2),'Position', [0.505 0.01 0.485 0.94]);
                imgdata = imread(MAPnamefull,SZ);
                imshow(imgdata,'border','tight');
                title(sprintf('Heatmap of angle, %dx%d, %d-bit ,%3.2fM,',IMGinfo(SZ).Width,...
                    MAPinfo(SZ).Height,MAPinfo(SZ).BitDepth,MAPinfo(SZ).FileSize/10^6),'fontweight','normal','FontSize',10)
                axis image
                linkaxes(axLINK1,'xy')
            end
            %histogram
              guiFig4 = findobj(0,'Tag','CA angle distribution');
            if isempty(guiFig4)
                guiFig4 = figure('Resize','on','Color',defaultBackground','Units','normalized',...
                    'Position',[0.258+0.474*ssU(4)/ssU(3)*2 0.308 0.30*ssU(4)/ssU(3) 0.35],...
                    'Visible','off','MenuBar','figure','Name','CA angle distribution','Tag', 'CA angle distribution',...
                    'NumberTitle','off','UserData',0); 
            end
            if exist(Output_values_name,'file')
                figure(guiFig4)
                htabgroup = uitabgroup(guiFig4);
                tabfig_name1 = uitab(htabgroup, 'Title','Angle-Histogram');
                hax_hist = axes('Parent', tabfig_name1);
                set(hax_hist,'Position',[0.15 0.15 0.80 0.80]);
                
                output_values = importdata(Output_values_name);
                if strcmp(CA_data_current{selectedROWs(1),8}, 'YES')  % with boundary
                    bins = 2.5:5:87.5;
                    hist(output_values(:,1),bins);
                    [n_ang xout_ang] = hist(output_values(:,1),bins);
                    xlim([0 90]);
                elseif strcmp(CA_data_current{selectedROWs(1),8}, 'NO')  % no boundary
                    bins = 2.5:5:177.5;
                    hist(output_values(:,1),bins);
                    [n_ang xout_ang] = hist(output_values(:,1),bins);
                    xlim([0 180]);
                end
                xlabel('Angle [degree]')
                ylabel('Frequency [#]')
                axis square
                %Compass plot
                tabfig_name2 = uitab(htabgroup, 'Title','Angle-Compass');
                hax_cmpp = axes('Parent', tabfig_name2);
                set(hax_cmpp,'Position',[0.10 0.10 0.80 0.80]);
                U = cosd(xout_ang).*n_ang;
                V = sind(xout_ang).*n_ang;
                compass(U,V)
            end
        end
    end
%--------------------------------------------------------------------------
% callback function for fiber analysis mode drop down
    function fibModeCallback(source,eventdata)
        str = get(source,'String');
        val = get(source,'Value');
        switch str{val};
            case 'CT'
                set(infoLabel,'String',note1);
                set(bndryModeDrop,'String',{'No Boundary','CSV Boundary','Tiff Boundary'});
                set(bndryModeDrop,'Value',1);
                fibMode = 0;
                bndryModeCallback(bndryModeDrop,0);
            case 'CT-FIRE Segments'
                set(infoLabel,'String',[note1 note2]);
                set(bndryModeDrop,'String',{'No Boundary','Tiff Boundary'});
                set(bndryModeDrop,'Value',1);
                fibMode = 1;
                bndryModeCallback(bndryModeDrop,0);
            case 'CT-FIRE Fibers'
                set(infoLabel,'String',[note1 note2]);
                set(bndryModeDrop,'String',{'No Boundary','Tiff Boundary'});
                set(bndryModeDrop,'Value',1);
                fibMode = 2;
                bndryModeCallback(bndryModeDrop,0);
            case 'CT-FIRE Endpoints'
                set(infoLabel,'String',[note1 note2]);
                set(bndryModeDrop,'String',{'No Boundary','Tiff Boundary'});
                set(bndryModeDrop,'Value',1);
                fibMode = 3;
                bndryModeCallback(bndryModeDrop,0);
        end
    end
%--------------------------------------------------------------------------
% callback function for boundary mode drop down
    function bndryModeCallback(source,eventdata)
        str = get(source,'String');
        val = get(source,'Value');
        switch str{val};
            case 'No Boundary'
                if fibMode == 0
                    set(infoLabel,'String',[note1]);
                else
                    set(infoLabel,'String',[note1 note2]);
                end
                bndryMode = 0;
            case 'Draw Boundary'
                if fibMode == 0
                    set(infoLabel,'String',[note1]);
                else
                    set(infoLabel,'String',[note1 note2]);
                end
                bndryMode = 1;
            case 'CSV Boundary'
                if fibMode == 0
                    set(infoLabel,'String',[note1 note3C note3]);
                else
                    set(infoLabel,'String',[note1 note2 note3c note3]);
                end
                bndryMode = 2;
            case 'Tiff Boundary'
                if fibMode == 0
                    set(infoLabel,'String',[note1 note3T note3]);
                else
                    set(infoLabel,'String',[note1 note2 note3T note3]);
                end
                bndryMode = 3;
        end
    end
%--------------------------------------------------------------------------
% callback function for imgOpen
    function getFile(imgOpen,eventdata)
        
        [fileName pathName] = uigetfile({'*.tif;*.tiff;*.jpg;*.jpeg';'*.*'},'Select Image',pathNameGlobal,'MultiSelect','on');
        if pathName ~= 0
            outDir = fullfile(pathName, 'CA_Out');   
            outDir2 = fullfile(pathName, 'CA_Boundary');   
        elseif pathName == 0
            disp('No file is selected')
            return;
        end
        if (~exist(outDir,'dir')||~exist(outDir2,'dir'))
            mkdir(outDir);mkdir(outDir2);
        end
        pathNameGlobal = pathName;
        save('currentP_CA.mat','pathNameGlobal','keepValGlobal','distValGlobal');
        %YL: define all the output files, directory here
        ROIimgDir = fullfile(pathName,'CA_ROI','Batch','ROI_analysis');
        ROIanaBatOutDir = fullfile(ROIimgDir,'CA_Out');
        ROImanDir = fullfile(pathName,'ROI_management');
        
        ROIanaDir = fullfile(pathName,'CA_ROI','Batch');
        ROIDir = fullfile(pathName,'CA_ROI');
                % folders for CA post ROI analysis of multiple(Batch-mode) images
        ROIpostBatDir = fullfile(pathName,'CA_ROI','Batch','ROI_post_analysis');
        BoundaryDir = fullfile(pathName,'CA_Boundary');
        if iscell(fileName) %check if multiple files were selected
            numFiles = length(fileName);
            disp(sprintf('%d files were selected',numFiles));
            set(imgLabel,'String',fileName);
            
            %open the first file for viewing 
            ff = fullfile(pathName,fileName{1});
            info = imfinfo(ff);
            numSections = numel(info);
            if numSections > 1
                img = imread(ff,1,'Info',info);
            else
                img = imread(ff);
            end
            figure(guiFig);
            if size(img,3) > 1
               if advancedOPT.plotrgbFLAG == 0
                   img = rgb2gray(img);
                   disp('color image was loaded but converted to grayscale image')
                   img = imadjust(img);
               elseif advancedOPT.plotrgbFLAG == 1
                   disp('display color image');
               end
            end
            imshow(img,'Parent',imgAx); hold on;
            set(guiFig,'name',sprintf('CurveAlign Figure: %s: first image of %d images',fileName{1},numFiles))
            imgSize(1,1) = size(img,1);
            imgSize(1,2) = size(img,2);
            %do not allow boundary drawing in batch mode
            if fibMode == 0 && bndryMode == 1 %CT only mode, and draw boundary
                disp('Cannot draw boundaries in batch mode.');
                set(infoLabel,'String','Cannot draw boundaries in batch mode.');
                return;
            end
        else
            numFiles = 1;
            set(imgLabel,'String',fileName);
            %open file for viewing
            ff = fullfile(pathName,fileName);
            info = imfinfo(ff);
            numSections = numel(info);
            if numSections > 1
                img = imread(ff,1,'Info',info);
                set(stackSlide,'max',numSections);
                set([stackSlide slideLab],'Enable','on');
                %                 set(wholeStack,'Enable','on');
                set(stackSlide,'SliderStep',[1/(numSections-1) 3/(numSections-1)]);
                set(stackSlide,'Callback',{@slider_chng_img});
                set(slideLab,'String','Stack image selected: 1');
            else
                img = imread(ff);
                set([stackSlide slideLab],'Enable','off')
            end
            if size(img,3) > 1
               if advancedOPT.plotrgbFLAG == 0
                   img = rgb2gray(img);
                   disp('color image was loaded but converted to grayscale image')
                   img = imadjust(img);
               elseif advancedOPT.plotrgbFLAG == 1
                   
                   disp('display color image');
                   
               end

            end
            if isempty(findobj(0,'-regexp','Name','CurveAlign Figure*'))
                guiFig = figure('Resize','on','Units','pixels','Position',guiFig_absPOS,...
                    'Visible','off','MenuBar','figure','name','CurveAlign Figure','NumberTitle','off','UserData',0);
                imgPanel = uipanel('Parent', guiFig,'Units','normalized','Position',[0 0 1 1]);
                imgAx = axes('Parent',imgPanel,'Units','normalized','Position',[0 0 1 1]);
            end
            figure(guiFig);
            imshow(img,'Parent',imgAx); 
            imgSize(1,1) = size(img,1);
            imgSize(1,2) = size(img,2);
            setappdata(imgOpen,'img',img);
            setappdata(imgOpen,'type',info(1).Format)
            colormap(gray);
            set(guiFig,'Visible','on');
            %Make filename to be a CELL array,
            % makes handling filenames more general, saves code.
            fileName = {fileName};
        end
        [~,~,fileEXT] = fileparts(fileName{1});
        %Give instructions about what to do next
        if fibMode == 0
            %CT only mode
            set(infoLabel,'String','Enter a coefficient threshold in the "keep" edit box. ');
            set([keepLab1],'ForegroundColor',[0 0 0])
            set(enterKeep,'Enable','on');
        else
            %CT-FIRE mode (in this mode, CT-FIRE output files must be present)
            %use default ctfire output folder to check the availability of
            %the mat files
            ctfFnd = checkCTFireFiles(fullfile(pathName,'ctFIREout'), fileName);
            if (~isempty(ctfFnd))
                set(infoLabel,'String','');
            else
                set(infoLabel,'String','One or more CT-FIRE files are missing.');
                return;
            end
        end
        str = get(infoLabel,'String'); %store whatever is the message so far, so we can add to it
        if bndryMode == 1
            %Alt click a boundary
            set(enterDistThresh,'Enable','on');
            set(infoLabel,'String',[str 'Alt-click a boundary. Enter distance value. Click Run.']);
        elseif bndryMode == 2 || bndryMode == 3
            %check to make sure the proper boundary files exist
            bndryFnd = checkBndryFiles(bndryMode, BoundaryDir, fileName);
            if (~isempty(bndryFnd))
                %Found all boundary files
                set(distLab,'ForegroundColor',[0 0 0]);
                set(enterDistThresh,'Enable','on');
                set(infoLabel,'String',[str 'Enter distance value. Click Run.']);
                set(makeAssoc,'Enable','on');
            else
                %Missing one or more boundary files
                set(infoLabel,'String',[str 'One or more boundary files are missing. Draw or add the boundary files to proceed']);
            end
        else
            %boundary mode = 0, no boundary
            set(infoLabel,'String',sprintf('Enter a coefficient threshold in the "keep" edit box then click Run for Curvelets based fiber feature extraction.\n OR click ROI Manager or ROI analysis for ROI-related operation'));
        end
        %if boundary file exists, overlay the boundary on the original image
         %Get the boundary data
         if (~isempty(bndryFnd))
            if bndryMode == 2
                figure(guiFig),hold on
                coords = csvread(fullfile(BoundaryDir,sprintf('boundary for %s.csv',fileName{1})));
                plot(coords(:,1),coords(:,2),'y','Parent',imgAx);
                plot(coords(:,1),coords(:,2),'*y','Parent',imgAx);
                hold off
            elseif bndryMode == 3
                figure(guiFig),hold on
                bff = fullfile(BoundaryDir, sprintf('mask for %s.tif',fileName{1}));
                bdryImg = imread(bff);
                [B,L] = bwboundaries(bdryImg,4);
                coords = B;%vertcat(B{:,1});
                for k = 1:length(coords)%2:length(coords)
                    boundary = coords{k};
                    plot(boundary(:,2), boundary(:,1), 'y','Parent',imgAx)
                end
                hold off
            end
         end
        M = size(img,1);
        N = size(img,2);
        advancedOPT.heatmap_STDfilter_size = ceil(N/32);  % YL: default value is consistent with the drwaMAP
        clear M N
        set(imgRun,'Callback',{@runMeasure});
        set([makeRecon makeAngle makeFeat makeOver makeMap imgRun advOptions],'Enable','on');
        set([CAroi_man_button CAroi_ana_button],'Enable','on');
        set([makeRecon makeAngle],'Enable','off') % yl,default output

        %disable method selection
        set(bndryModeDrop,'Enable','off');
        set(fibModeDrop,'Enable','off');
         % add an option to show the previous analysis results
         % in "CA_Out" folder
         CAout_found = checkCAoutput(pathName,fileName);
         existing_ind = find(cellfun(@isempty, CAout_found) == 0); % index of images with existing output
         if isempty(existing_ind)
             disp('No previous analysis was found.')
         else
           disp(sprintf('Previous CurveAlign analysis was found for %d out of %d opened image(s)',...
                 length(existing_ind),length(fileName)))
             % user choose to check the previous analysis
             choice=questdlg('Check previoius CurveAlign results?','Previous CurveAlign analysis exists','Yes','No','Yes');
             if(isempty(choice))
                 return;
             else
                 switch choice
                     case 'Yes'
                         set(infoLabel, 'String',sprintf('Existing CurveAlign results listed:%d out of %d opened image(s) \n Running "CurveAlign" here will overwrite them. ',...
                       length(existing_ind),length(fileName)))
                         checkCAout_display_fn(pathName,fileName,existing_ind);
                     case 'No'
                         set(infoLabel,'String','Choose "RUN options"  and if necesssary, set/confirm parameters')   
                         return;
                 end
             end
         end
    end
%--------------------------------------------------------------------------
% callback function for listbox 'imgLabel'
    function imgLabel_Callback(imgLabel, eventdata, handles)
        % hObject    handle to imgLabel
        % eventdata  reserved - to be defined in a future version of MATLAB
        % handles    structure with handles and user data (see GUIDATA)
        % Hints: contents = cellstr(get(hObject,'String')) returns contents
        % contents{get(hObject,'Value')} returns selected item from listbox1
        items = get(imgLabel,'String');
        if ~iscell(items)
            items = {items};
        end
        if strcmp(items{1},'None Selected')
            error('No image is opened')
        end
        index_selected = get(imgLabel,'Value');
        item_selected = items{index_selected};
%         display(item_selected);
        item_fullpath = fullfile(pathName,item_selected);
        iteminfo = imfinfo(item_fullpath);
        item_numSections = numel(iteminfo);
        ff = item_fullpath; info = iteminfo; numSections = item_numSections;
            if item_numSections > 1
                img = imread(item_fullpath,1,'Info',info);
                set(stackSlide,'max',item_numSections);
                set([stackSlide slideLab],'Enable','on');
                set(stackSlide,'SliderStep',[1/(item_numSections-1) 3/(item_numSections-1)]);
                set(slideLab,'String','Stack image selected: 1');
            else
                img = imread(item_fullpath);
                set([stackSlide slideLab],'Enable','off');
            end
            if size(img,3) > 1
                if advancedOPT.plotrgbFLAG == 0
                    img = rgb2gray(img);
                    disp('color image was loaded but converted to grayscale image')
                    img = imadjust(img);
                elseif advancedOPT.plotrgbFLAG == 1
                    disp('display color image');
                end
            end
            if isempty(findobj(0,'-regexp','Name','CurveAlign Figure*'))
                guiFig = figure('Resize','on','Units','pixels','Position',guiFig_absPOS,...
                    'Visible','off','MenuBar','figure','name','CurveAlign Figure','NumberTitle','off','UserData',0,...
                    'KeyPressFcn',@roi_mang_keypress_fn);
                imgPanel = uipanel('Parent', guiFig,'Units','normalized','Position',[0 0 1 1]);
                imgAx = axes('Parent',imgPanel,'Units','normalized','Position',[0 0 1 1]);
            end
            figure(guiFig); %set(imgAx,'NextPlot','add');
%             set(imgAx,'NextPlot','new');
            set(imgAx,'NextPlot','replace');
%             img = imadjust(img);
            imshow(img,'Parent',imgAx); 
            imgSize(1,1) = size(img,1);
            imgSize(1,2) = size(img,2);
           if item_numSections == 1
               set(guiFig,'name',sprintf('CurveAlign Figure: %s, %dx%d pixels, %d-bit',item_selected,info.Height,info.Width,info.BitDepth))
           elseif item_numSections > 1   % stack
               set(guiFig,'name',sprintf('CurveAlign Figure: (1/%d)%s, %dx%d pixels, %d-bit stack',item_numSections,item_selected,info(1).Height,info(1).Width,info(1).BitDepth))
           end
           % if csv or tif boundary exists, overlay it on the original image
           if bndryMode >= 1
               bndryFnd = checkBndryFiles(bndryMode, BoundaryDir, {item_selected});
               if (~isempty(bndryFnd))
                   if bndryMode == 1 || bndryMode == 2
                       figure(guiFig),hold on
                       coords = csvread(fullfile(pathName,'CA_Boundary', sprintf('boundary for %s.csv',item_selected)));
                       plot(coords(:,1),coords(:,2),'y','Parent',imgAx);
                       plot(coords(:,1),coords(:,2),'*y','Parent',imgAx);
                       hold off
                   elseif bndryMode == 3
                       figure(guiFig),hold on
                       bff = fullfile(pathName,'CA_Boundary',sprintf('mask for %s.tif',item_selected));
                       bdryImg = imread(bff);
                       [B,L] = bwboundaries(bdryImg,4);
                       coords = B;%vertcat(B{:,1});
                       for k = 1:length(coords)%2:length(coords)
                           boundary = coords{k};
                           plot(boundary(:,2), boundary(:,1), 'y','Parent',imgAx)
                       end
                       hold off
                   end
               end
           end
            setappdata(imgOpen,'img',img);
            setappdata(imgOpen,'type',info(1).Format)
            set(guiFig,'Visible','on');
            M = size(img,1);
            N = size(img,2);
            advancedOPT.heatmap_STDfilter_size = ceil(N/32);  % YL: default value is consistent with the drwaMAP
            clear M N
    end
%%-------------------------------------------------------------------------
%call back function for push button CTFIRE_Callback
    function CTFIRE_Callback(hObject,eventdata)
        ca2ctf.flag = 1;
        ctFIRE(ca2ctf);
        return
    end
%%-------------------------------------------------------------------------
%call back function for push button BDcsv_Callback
    function BDcsv_Callback(hObject,eventdata)
        figure(guiFig);
        set(infoLabel,'String',sprintf('Alt-click to draw a csv boundary for %s. To finish, release Alt.',fileName{index_selected}));
        
        set(guiFig,'UserData',0)
        
        if ~get(guiFig,'UserData')
            set(guiFig,'WindowKeyPressFcn',@startPoint)
            coords = [-1000 -1000];
            aa = 1;
        end
    end
%--------------------------------------------------------------------------
%callback function for push button
    function BDmask_Callback(hObject,eventdata)
        
        if BDCchoice == 1 & length(fileName) > 1  % for batch-mode manual tiff BD creation
            
            disp('Using the same settings to manually draw the tiff boundary in batch mode')
            
        elseif BDCchoice == 3 & length(fileName) > 1  % for batch-mode manual csv BD creation
            disp('Using the same settings to manually draw the csv boundary in batch mode')
            BDcsv_Callback
            return
        elseif isempty(BDCchoice) & length(fileName)== 0
            figure(BDCgcf);
            set(BDCgcf,'Visible', 'on')
            disp('No image is open. Launch automatic boundary creation module to automatically segment boundary based on HE image');
            return
        else
            BWmaskChoice1 = questdlg('Manual or automatic boundary creation ?', ...
                'Boundary creation options','Manual Mask','Automatic Mask','Manual csvBD','Manual Mask');
            switch BWmaskChoice1
                case 'Manual Mask'
                    BDCchoice = 1;      %
                    disp('manually draw boundary on opened SHG image')
                case 'Automatic Mask'
                    BDCchoice = 2;     %
                    disp('Automatically segment boundary based on HE image');
                case 'Manual csvBD'
                    BDCchoice = 3;     
                    if isempty(img)
                        disp('No image is opened to manually anntotate the points on a boundary.')
                    else
                       BDcsv_Callback
                    end%
                    return
            end
            if BDCchoice == 2
                figure(BDCgcf);
                set(BDCgcf,'Visible', 'on')
                return
            end
            BWmaskChoice = questdlg('draw boundary with freehand or polygon?', ...
                'Manually draw boundary','freehand mode','polygon mode','freehand mode');
            BW_shape = [];
            switch BWmaskChoice
                case 'freehand mode'
                    BW_shape = 2;      % freehand, consistent with ROI_shape
                    set(infoLabel,'String','Click on the image to start freehand boundary creation')
                    disp('draw freehand boundary')
                case 'polygon mode'
                    BW_shape = 4;      % polygon, consistent with ROI_shape
                    set(infoLabel,'String','Click on the image to start polygon boundary creation')
                    disp('draw polygon boundary');
            end
        end
        double_click = 0;  % YL
        figure(guiFig);
        if isempty(BW_shape)
            disp('please choose the right Boundary shape')
            return
        end
        if BW_shape == 2;     
            set(infoLabel,'String','Click on the image to start freehand boundary creation')
        elseif BW_shape == 4;
            set(infoLabel,'String','Click on the image to start polygon boundary creation')
        end
        g_mask=logical(0);
        while(double_click==0)
            if (BW_shape == 2)
                maskh = imfreehand;
                set(infoLabel,'String','Drawing boundary..., To finish, press "m" and then click any point on the image');
            elseif (BW_shape == 4)
                maskh = impoly;
                set(infoLabel,'String','Drawing boundary..., To finish,  press "m" and then double click any point on the image');
            end
            MaskB= createMask(maskh);
            g_mask=g_mask|MaskB;
            figure(guiFig);
        end
        BDmaskname = fullfile(BoundaryDir,sprintf('mask for %s.tif',fileName{index_selected}));
        if ~exist(BoundaryDir,'dir')
            mkdir(BoundaryDir);
        end
        imwrite(g_mask,BDmaskname,'Compression','none')
        %donot enable "imRun" after mask create mask
        set([imgRun makeAngle makeRecon enterKeep enterDistThresh makeOver makeMap makeFeat],'Enable','off')
        set([makeRecon makeAngle],'Value',3)
        set(infoLabel,'String',sprintf('tiff mask was created for %s. To use this mask: Reset and set boundary mode to tiff boundary',fileName{index_selected}));
        fprintf('Tiff mask is saved at %s \n',BDmaskname)
    end
% callback function for HEfileopen
    function getHEfiles_Callback(hObject,eventdata)
       [HEfilename,HEpathname] = uigetfile({'*.tif;*.tiff;*.jpg';'*.*'},'Select HE color Image',HEpathname,'MultiSelect','on');
       if  HEpathname == 0
           HEpathname = '';
           HEfilename = '';
           disp('No HE file is selected')
           return
       else
       set(HEfileinfo,'String',HEpathname)
       if ~iscell(HEfilename)
           HEfilename = {HEfilename}
       end
        BDCparameters.HEfilename = HEfilename;
        BDCparameters.HEfilepath = HEpathname;
        if get(HEreg_FLAG,'Value') == 3
            BDC_Operation_name = 'registration';
        else
            BDC_Operation_name = 'segmentation';
        end
        if length(HEfilename) == 1
            disp(sprintf('%d HE file is opened from %s for %s',length(HEfilename),HEpathname, BDC_Operation_name));
            disp(sprintf('Imge name is %s', HEfilename{1}))
        else
            disp(sprintf('%d HE files are opened from %s for %s',length(HEfilename),HEpathname,BDC_Operation_name))
            for i = 1:length(HEfilename)
                disp(sprintf('%d: %s',i,HEfilename{i})) 
            end
        end
       end
    end

% callback function for HEfileopen
   function enterHEfolder_Callback(hObject,eventdata)
       HEpathname = get(HEfileinfo,'String');
       getHEfiles_Callback
    end
%--------------------------------------------------------------------------
% callback function for SHGfileopen
    function getSHGfolder_Callback(hObject,eventdata)
       
       SHGpathname = uigetdir(SHGpathname,'Selected SHG image folder');
       if  SHGpathname == 0
           SHGpathname = '';
           disp('No SHG folder is selected.')
           return
           
       else
           BDCparameters.SHGfilepath = SHGpathname;
           set(SHGfolderinfo,'String',SHGpathname)
           if get(HEseg_FLAG, 'Value') == 3    % for segmentation
              sprintf( 'The Segmented boundary file will be saved at %s', fullfile(SHGpathname,'CA_Boundary'))
           elseif get(HEreg_FLAG, 'Value') == 3    % for registration
               ii = 0;
               for i = 1:length(HEfilename)
                   if ~exist(fullfile(SHGpathname, HEfilename{i}),'file')
                       disp(sprintf('SHG file not exist for %s',HEfilename{i}))
                   else
                       ii = ii + 1;
                   end
               end
               if ii == length(HEfilename)
                  disp(sprintf('All associated %d SHG file(s) were found',length(HEfilename)))
               else
                   disp(sprintf('%Missing %d associated SHG files.',length(HEfilename)-ii)) 
               end
              
           end
       end
   
    end
%--------------------------------------------------------------------------
% callback function for infileopen
    function enterSHGfolder_Callback(~,~)
       SHGpathname = get(SHGfolderinfo,'String');
       getSHGfolder_Callback
    end
%--------------------------------------------------------------------------
% callback function for HE_RES_edit_Callback text box
    function HE_RES_edit_Callback(hObject,eventdata)
        usr_input = get(HE_RES_edit,'String');
        usr_input = str2double(usr_input);
        set(HE_RES_edit,'UserData',usr_input);
        pixelpermicron = usr_input;
        BDCparameters.pixelpermicron = pixelpermicron;
        disp(sprintf('Pixel per micron ratio is set to %3.2f',pixelpermicron))
    end
%--------------------------------------------------------------------------
% callback function for HE_threshold_eidt_Callback text box
    function HE_threshold_edit_Callback(hObject,eventdata)
        usr_input = get(HE_threshold_edit,'String');
        usr_input = str2double(usr_input);
        set(HE_threshold_edit,'UserData',usr_input);
        areaThreshold = usr_input;
        BDCparameters.areaThreshold = areaThreshold;
        disp(sprintf('Area threshold is set to %5.0f',areaThreshold))
    end

%--------------------------------------------------------------------------
% callback function for HE_threshold_eidt_Callback text box
    function HEreg_FLAG_Callback(hObject,eventdata)
        if get(HEreg_FLAG,'Value') == 0
            set(HEseg_FLAG,'Value',3)
        elseif get(HEreg_FLAG,'Value') == 3
            set(HEseg_FLAG,'Value',0)
        end
    end
%--------------------------------------------------------------------------
% callback function for HE_threshold_eidt_Callback text box
    function HEseg_FLAG_Callback(hObject,eventdata)
        if get(HEseg_FLAG,'Value') == 3
            set(HEreg_FLAG,'Value',0)
        elseif get(HEseg_FLAG,'Value') == 0
            set(HEreg_FLAG,'Value',3)
        end
    end
%--------------------------------------------------------------------------
% callback function for HE_threshold_eidt_Callback text box
    function BDCgcfOK_Callback(hObject,eventdata)
        if isempty(BDCparameters.HEfilename) 
            disp('HE file(s) not selected')
            return
        end
        if isempty(BDCparameters.SHGfilepath)             
            disp('SHG folder not selected')
            return
        end
        areaThreshold = str2num(get(HE_threshold_edit,'String'));
        BDCparameters.areaThreshold = areaThreshold;
        pixelpermicron = str2num(get(HE_RES_edit,'String'));
        BDCparameters.pixelpermicron = pixelpermicron;
        BDCparametersTEMP = BDCparameters;
        if get(HEseg_FLAG,'Value') == 3 & get(HEreg_FLAG,'Value') == 0    % segmentation
            for i = 1:length(BDCparameters.HEfilename)
                try
                    BDCparametersTEMP.HEfilename = BDCparameters.HEfilename{i};
                    disp(sprintf('Mask creation is in progress...: %d/%d from %s',i, length(BDCparameters.HEfilename),BDCparameters.HEfilename{i}))
                    tic;
                    I = BDcreationHE(BDCparametersTEMP);
                    disp(sprintf('takes %4.3f seconds on %s', toc, BDCparametersTEMP.HEfilename))
                    figure('pos', [200+50*i 200+25*i ssU(4) ssU(4)/3],'name',BDCparametersTEMP.HEfilename,'NumberTitle','off');
                    HEdata = imread(fullfile(BDCparameters.HEfilepath, BDCparameters.HEfilename{i}));
                    ax(1) = subplot(1,3,1); imshow(HEdata); title('HE image registered');   
                    ax(2) = subplot(1,3,2); imshow(I); title('Segmentation'); 
                    ax(3) = subplot(1,3,3); imshowpair(I,HEdata);title('Difference image')
                    linkaxes(ax,'xy')
                    drawnow
                catch HEsegErr
                    disp(sprintf('Error message for the segmentation of %s: %s',...
                    BDCparametersTEMP.HEfilename, HEsegErr.message))
                end
                    
            end
        elseif get(HEseg_FLAG,'Value') == 0 & get(HEreg_FLAG,'Value') == 3 % registration
            
            for i = 1:length(BDCparameters.HEfilename)
                try
                    BDCparametersTEMP.HEfilename = BDCparameters.HEfilename{i};
                    disp(sprintf('Registration is in progress...: %d/%d from %s',i, length(BDCparameters.HEfilename),BDCparameters.HEfilename{i}))
                    tic;
%                     I = BDcreation_reg(BDCparametersTEMP);
                    I = BDcreation_reg_modified(BDCparametersTEMP);
                    disp(sprintf('takes %4.3f seconds on %s', toc, BDCparametersTEMP.HEfilename))
%                     figure('pos', [200+50*i 200+25*i ssU(4) ssU(4)/3],'name',BDCparametersTEMP.HEfilename,'NumberTitle','off' );
%                     HEdata = imread(fullfile(BDCparameters.HEfilepath, BDCparameters.HEfilename{i}));
%                     SHGdata = imread(fullfile(BDCparameters.SHGfilepath, BDCparameters.HEfilename{i}));
%                     ax(1) = subplot(1,3,1); imshow(HEdata);title('original HE image');
%                     ax(2) = subplot(1,3,2); imshow(I); title('registered HE image');
%                     ax(3) = subplot(1,3,3); imshow(SHGdata);title('Corresponding SHG image');
%                     linkaxes(ax(2:3),'xy')
%                     drawnow
                catch HEregErr
                    disp(sprintf('Error message for the registration of %s: %s',...
                    BDCparametersTEMP.HEfilename, HEregErr.message))
                end
            end
        end
              
        set(BDCgcf,'Visible', 'off')
        disp(sprintf('Automatic boundary creation from %d HE images is done.',length(BDCparameters.HEfilename)))
        
      end
%-----------------------------------------------------------------------
% callback function for HE_threshold_eidt_Callback text box
    function BDCgcfCANCEL_Callback(hObject,eventdata)
        
        set(BDCgcf,'Visible', 'off')
        disp('Automatic boundary creation is cancelled ')
  
    end
%--------------------------------------------------------------------------
% callback function for stack slider
    function slider_chng_img(hObject,eventdata)
        idx = round(get(hObject,'Value'));
        img = imread(ff,idx,'Info',info);
        [~,tempname,tempext] = fileparts(ff);
        item_selected = strcat(tempname,tempext);
        set(imgAx,'NextPlot','new');
%         img = imadjust(img);  %YL
        imshow(img,'Parent',imgAx);
        set(guiFig,'name',sprintf('(%d/%d)%s, %dx%d pixels, %d-bit stack',idx,numSections,item_selected,info(idx).Height,info(idx).Width,info(idx).BitDepth))
        set(imgAx,'NextPlot','add');
        if ~isempty(coords) %if there is a boundary, draw it now
            if bndryMode == 2
                plot(coords(:,1),coords(:,2),'y','Parent',imgAx);
                plot(coords(:,1),coords(:,2),'*y','Parent',imgAx);
            elseif bndryMode == 3
                for k = 1:length(coords)%2:length(coords)
                    boundary = coords{k};
                    plot(boundary(:,2), boundary(:,1), 'y','Parent',imgAx)
             
                end
            end
        end
        setappdata(imgOpen,'img',img);
       set(slideLab,'String',['Stack image selected: ' num2str(idx)]);
    end
%--------------------------------------------------------------------------
% callback function for enterKeep text box
    function get_textbox_data(enterKeep,eventdata)
        usr_input = get(enterKeep,'String');
        usr_input = str2double(usr_input);
        set(enterKeep,'UserData',usr_input)
    end
%--------------------------------------------------------------------------
% callback function for enterDistThresh text box
    function get_textbox_data2(enterDistThresh,eventdata)
        usr_input = get(enterDistThresh,'String');
        usr_input = str2double(usr_input);
        set(enterDistThresh,'UserData',usr_input)
    end
%%--------------------------------------------------------------------------
%%callback function for CAroi button
   function CAroi_man_Callback(CAroibutton,evendata)
     %% Option for ROI manager
     % save current parameters
       outDir = fullfile(pathName,'CA_Out');
        if ~exist(outDir,'dir')
            mkdir(outDir);
        end
        outDir = fullfile(pathName,'CA_Boundary');
        
        %         IMG = getappdata(imgOpen,'img');
        keep = get(enterKeep,'UserData');
        distThresh = get(enterDistThresh,'UserData');
        keepValGlobal = keep;
        distValGlobal = distThresh;
        save('currentP_CA.mat','pathNameGlobal','keepValGlobal','distValGlobal');
        if isempty(keep)
            %indicates the % of curvelets to process (after sorting by
            %coefficient value)
            keep = .001;
        end
        
        if isempty(distThresh)
            %this is default and is in pixels
            distThresh = 100;
        end
        
        if bndryMode == 2 || bndryMode == 3
            setappdata(guiFig,'boundary',1)
        elseif bndryMode == 0
            coords = []; %no boundary
            bdryImg = [];
        else
            disp(sprintf('csv boundary file name: boundary for %s.csv',fileName{index_selected}))
        end
        %check if user directed to output boundary association lines (where
        %on the boundary the curvelet is being compared)
        makeAssocFlag = get(makeAssoc,'Value') == get(makeAssoc,'Max');
        
        makeFeatFlag = get(makeFeat,'Value') == get(makeFeat,'Max');
        makeOverFlag = get(makeOver,'Value') == get(makeOver,'Max');
        makeMapFlag = get(makeMap,'Value') == get(makeMap,'Max');
        
      save(fullfile(pathName,'currentP_CA.mat'),'keep', 'coords', 'distThresh', 'makeAssocFlag', 'makeMapFlag', 'makeOverFlag', 'makeFeatFlag', 'infoLabel', 'bndryMode', 'bdryImg', 'pathName', 'fibMode','numSections','advancedOPT');
      CAcontrol.imgAx = imgAx; 
      CAcontrol.idx = idx; 
      CAcontrol.fibMode = fibMode; 
      CAcontrol.plotrgbFLAG = advancedOPT.plotrgbFLAG;
      CAcontrol.specifyROIsize = advancedOPT.specifyROIsize;
      CAcontrol.fiber_midpointEST = advancedOPT.fiber_midpointEST;
      CAcontrol.loadROIFLAG = loadROIFLAG;
      CAcontrol.guiFig_absPOS = guiFig_absPOS;
      if CAcontrol.loadROIFLAG == 1
         CAcontrol.roiMATnamefull = roiMATnameV{index_selected};
         CAcontrol.folderROIman = advancedOPT.folderROIman;
      else
          CAcontrol.roiMATnamefull = '';
          CAcontrol.folderROIman = '';
      end
      CAroi(pathName,fileName{index_selected},[],CAcontrol);
   end

%%--------------------------------------------------------------------------
%%callback function for CAroi button
   function CAroi_ana_Callback(hobject,evendata)
     % Option for ROI analysis
     % save current parameters
         set(infoLabel,'String', 'Start CurveAlign ROI analysis for the ROIs defined by ROI manager') 
         if fibMode == 0    % CT-mode
             ROIanaChoice = questdlg('ROI analysis for post-processing or on cropped ROI image or ROI mask?', ...
                 'ROI analysis','ROI post-processing','CA on cropped rectanglar ROI','CA on mask with ROI of any shape','ROI post-processing');
             if isempty(ROIanaChoice)
                 error('choose the ROI analysis mode to proceed')
             end
             switch ROIanaChoice
                 case 'ROI post-processing'
                     if numSections > 1    
                        disp('ROI post-processing on stack')
                     end
                     postFLAG = 1;
                     cropIMGon = 0;
                     disp('ROI Post-processing on the CA features')
                 case 'CA on cropped rectanglar ROI'
                     postFLAG = 0;
                     cropIMGon = 1;
                     disp('CA alignment analysis on the the cropped rectangular ROIs')
                 case 'CA on mask with ROI of any shape'
                     postFLAG = 0;
                     cropIMGon = 0;
                     disp('CA alignment analysis on the the ROI mask of any shape');
             end
         else
             postFLAG = 1;
             cropIMGon = 0;
         end
         
         if postFLAG == 1
             % Check the previous CA analysis results as well as the running
             % parameters
             ii = 0;  % count the number of files that are not processed with the same fiber mode or boundary mode
             jj = 0;  % count the number of all the output mat files
             for i = 1:length(fileName)
                 [~,fileNameNE,fileEXT] = fileparts(fileName{i}) ;
                 for j = 1:numSections
                     jj = jj + 1;
                     if numSections > 1
                         filename_temp = [fileNameNE sprintf('_s%d',j) '.tif'];
                         matfilename = [fileNameNE sprintf('_s%d',j) '_fibFeatures'  '.mat'];
                     elseif numSections == 1
                         filename_temp = fileName{i};
                         matfilename = [fileNameNE '_fibFeatures'  '.mat'];
                     end
                     if exist(fullfile(pathName,'CA_Out',matfilename),'file')
                         matdata_CApost = load(fullfile(pathName,'CA_Out',matfilename),'tifBoundary','fibProcMeth');
                         if matdata_CApost.fibProcMeth ~=  fibMode || matdata_CApost.tifBoundary ~=  bndryMode;
                             ii = ii + 1;
                             disp(sprintf('%d: %s has NOT been analyzed with the specified fiber mode or boundary mode.',ii,fileNameNE))
                         end
                     else
                         ii = ii + 1;
                         disp(sprintf('%d: %s does NOT exist',ii,fullfile(pathName,'CA_Out',matfilename)))
                     end
                 end
             end
             if ii == 0
                 note_temp = 'previous full-size image analysis with the specified fiber and boundary mode exists';
                 disp(sprintf(' All %d %s ',jj, note_temp))
                 pause(1.5)
             elseif ii > 0
                 note_temp1 = 'does NOT have  previous full-size image analysis with the specified fiber and boundary mode';
                 note_temp2 = 'Prepare the full-size results before ROI post-processing';
                 set(infoLabel,'String',sprintf(' %d of %d %s. \n %s',ii,jj,note_temp1,note_temp2))
                 return
             end
         end
        
        CA_data_current = [];
        k = 0;
        for i = 1:length(fileName)
            [~,fileNameNE,fileEXT] = fileparts(fileName{i}) ;
            roiMATnamefull = [fileNameNE,'_ROIs.mat'];
            if exist(fullfile(ROImanDir,roiMATnamefull),'file')
                k = k + 1; disp(sprintf('Found ROI for %s',fileName{i}))
            else
                disp(sprintf('ROI for %s not exist',fileName{i}));
            end
        end
        if k ~= length(fileName)
            error(sprintf('Missing %d ROI files',length(fileName) - k)) 
        end
        if(exist(ROIanaBatOutDir,'dir')==0)%check for ROI folder
               mkdir(ROIanaBatOutDir);
        end
        % YL: get/load processing parameters
        if postFLAG == 0
            %         IMG = getappdata(imgOpen,'img');
            keep = get(enterKeep,'UserData');
            distThresh = get(enterDistThresh,'UserData');
            keepValGlobal = keep;
            distValGlobal = distThresh;
            save('currentP_CA.mat','pathNameGlobal','keepValGlobal','distValGlobal');
            if isempty(keep)
                %indicates the % of curvelets to process (after sorting by
                %coefficient value)
                keep = .001;
            end
            if isempty(distThresh)
                %this is default and is in pixels
                distThresh = 100;
            end
            if bndryMode == 2 || bndryMode == 3
                setappdata(guiFig,'boundary',1)
            elseif bndryMode == 0
                coords = []; %no boundary
                bdryImg = [];
            else
                disp(sprintf('csv boundary file name: boundary for %s.csv',fileName{index_selected}))
            end
            makeAssocFlag = get(makeAssoc,'Value') == get(makeAssoc,'Max');
            set(makeFeat,'Enable','off');
            set(makeOver,'Enable','off');
            set(makeMap,'Enable','off');
            makeFeatFlag = get(makeFeat,'Value') == get(makeFeat,'Max');
            makeOverFlag = get(makeOver,'Value') == get(makeOver,'Max');
            makeMapFlag = get(makeMap,'Value') == get(makeMap,'Max');
            save(fullfile(pathName,'currentP_CA.mat'),'keep', 'coords', 'distThresh', 'makeAssocFlag', 'makeMapFlag', 'makeOverFlag', 'makeFeatFlag', 'infoLabel', 'bndryMode', 'bdryImg', 'pathName', 'fibMode','numSections','advancedOPT');
        elseif postFLAG == 1   % post-processing of the CA features
            if(exist(ROIpostBatDir,'dir')==0)%check for ROI folder
                mkdir(ROIpostBatDir);
            end
        end
        % check the availability of output table
        CA_tablefig_find = findobj(0,'Name', 'CurveAlign output table');
        if isempty(CA_tablefig_find)
            
            CA_table_fig = figure('Units','normalized','Position',figPOS,'Visible','off',...
                'NumberTitle','off','name','CurveAlign output table');
            CA_output_table = uitable('Parent',CA_table_fig,'Units','normalized','Position',[0.05 0.05 0.9 0.9],...
                'Data', CA_data_current,...
                'ColumnName', columnname,...
                'ColumnFormat', columnformat,...
                'ColumnWidth',columnwidth,...
                'ColumnEditable', [false false false false false false false false false false false false false false],...
                'RowName',[],...
                'CellSelectionCallback',{@CAot_CellSelectionCallback});
        end
        % end of output table check
       items_number_current = 0;
       for i = 1:length(fileName)
           [~,fileNameNE,fileEXT] = fileparts(fileName{i}) ;
           roiMATnamefull = [fileNameNE,'_ROIs.mat'];
           try
               load(fullfile(ROImanDir,roiMATnamefull),'separate_rois')
               if isempty(separate_rois)
                   disp(sprintf('%s is empty. %s is skipped',fullfile(ROImanDir,roiMATnamefull),fileName{i}))
                   continue
               end
           catch exp_temp
               disp(sprintf('Error in loading %s: %s. %s is skipped',fullfile(ROImanDir,roiMATnamefull),exp_temp.message,fileName{i}))
               continue
           end
           ROInames = fieldnames(separate_rois);
           s_roi_num = length(ROInames);
           IMGname = fullfile(pathName,fileName{i});
           IMGinfo = imfinfo(IMGname);
           numSections = numel(IMGinfo); % number of sections, default: 1;
           BWcell = bdryImg;    % boundary for the full size image          
           ROIbw = BWcell;  %  for the full size image
           for j = 1:numSections
               if postFLAG == 1
                   if numSections > 1
                       matfilename = [fileNameNE sprintf('_s%d',j) '_fibFeatures'  '.mat'];
                       IMG = imread(IMGname,j);
                       IMGctf = fullfile(pathName,'ctFIREout',['OL_ctFIRE_',fileNameNE sprintf('_s%d',j) '.tif']);
                   elseif numSections == 1
                       matfilename = [fileNameNE '_fibFeatures'  '.mat'];
                       IMG = imread(IMGname);
                       IMGctf = fullfile(pathName,'ctFIREout',['OL_ctFIRE_',fileNameNE,'.tif']);  % CT-FIRE overlay 
                   end
                   if(exist(fullfile(pathName,'CA_Out',matfilename),'file')~=0)%~=0 instead of ==1 because value is equal to 2
                       matdata_CApost = load(fullfile(pathName,'CA_Out',matfilename),'fibFeat','tifBoundary','fibProcMeth','distThresh','coords');
                       fibFeat_load = matdata_CApost.fibFeat;
                       distThresh = matdata_CApost.distThresh;
                       tifBoundary = matdata_CApost.tifBoundary;  % 1,2,3: with boundary; 0: no boundary 
                       % load running parameters from the saved file
                       bndryMode = tifBoundary; 
                       coords = matdata_CApost.coords;
                       fibProcMeth = matdata_CApost.fibProcMeth; % 0: curvelets; 1,2,3: CTF fibers
                       fibMode = fibProcMeth;
                       cropIMGon = 0; 
                       cropFLAG = 'NO';                 % analysis based on orignal full image analysis
                       if fibMode == 0 % "curvelets"
                           modeID = 'Curvelets';
                       else %"CTF fibers" 1,2,3
                           modeID = 'CTF Fibers';
                       end
                       if bndryMode == 0
                           bndryID = 'NO';
                       elseif bndryMode == 2 || bndryMode == 3
                           bndryID = 'YES';
                       end
                       postFLAGt = 'YES';
                       try
                           % load the overlay image
                           if numSections > 1
                               overIMG = imread(fullfile(pathName,'CA_Out',[fileNameNE,'_overlay.tiff']),j);
                           elseif numSections == 1
                               overIMG = imread(fullfile(pathName,'CA_Out',[fileNameNE,'_overlay.tiff']));
                           end
                           figure(guiFig); set(imgAx,'NextPlot','replace');
                           set(guiFig,'Name',fileNameNE);
                           imshow(overIMG,'Parent',imgAx);hold on;
                           OLexistflag = 1;
                       catch
                           OLexistflag = 0;
                           try
                               disp(sprintf('%s does not exist \n display the CT-FIRE overlay image instead',fullfile(pathName,'CA_Out',[fileNameNE,'_overlay.tiff'])))
                               figure(guiFig); set(imgAx,'NextPlot','replace');
                               set(guiFig,'Name',fileNameNE);
                               imshow(IMGctf,'Parent',imgAx);hold on;
                           catch
                               disp(sprintf('%s does not exist \n dislpay the original image instead',fullfile(pathName,'CA_Out',[fileNameNE,'_overlay.tiff'])))
                               figure(guiFig); set(imgAx,'NextPlot','replace');
                               set(guiFig,'Name',fileNameNE);
                               imshow(IMG,'Parent',imgAx);hold on;
                           end
                       end
                   else
                       error(sprintf('CurveAlign feature file %s does not exist.', fullfile(pathName,'CA_Out',matfilename)));
                   end
               end
               if numSections == 1
                   IMG = imread(IMGname);
               else
                   IMG = imread(IMGname,j);
               end
               if size(IMG,3) > 1
                   if advancedOPT.plotrgbFLAG == 0
                       IMG = rgb2gray(IMG);
                       disp('color image was loaded but converted to grayscale image')
                   elseif advancedOPT.plotrgbFLAG == 1
                       disp('display color image');
                   end
               end
               for k=1:s_roi_num
                   if numSections == 1
                       set(infoLabel,'String',sprintf('Image %d/%d: ROI analysis on %s of %s',i, length(fileName),ROInames{k},fileName{i}))
                   else
                       set(infoLabel,'String',sprintf('Stack %d/%d, slice %d/%d, ROI analysis on %s of %s',...
                           i, length(fileName),j, numSections,ROInames{k},fileName{i}))
                   end
                   items_number_current = items_number_current+1;
                   ROIshape_ind = separate_rois.(ROInames{k}).shape;
                        if cropIMGon == 0     % use ROI mask
                            if   ~iscell(separate_rois.(ROInames{k}).shape)
                                ROIshape_ind = separate_rois.(ROInames{k}).shape;
                                BD_temp = separate_rois.(ROInames{k}).boundary;
                                boundary = BD_temp{1};
                                BW = roipoly(IMG,boundary(:,2),boundary(:,1));
                                yc = separate_rois.(ROInames{k}).xm;
                                xc = separate_rois.(ROInames{k}).ym;
                                z = j;
                            elseif iscell(separate_rois.(ROInames{k}).shape)
                                ROIshape_ind = nan;
                                s_subcomps=size(separate_rois.(ROInames{k}).shape,2);
                                s1=size(IMG,1);s2=size(IMG,2);
                                BW(1:s1,1:s2)=logical(0);
                                for m=1:s_subcomps
                                    boundary = cell2mat(separate_rois.(ROInames{k}).boundary{m});
                                    BW2 = roipoly(IMG,boundary(:,2),boundary(:,1));
                                    BW=BW|BW2;
                                end
                                xc = nan; yc = nan; z = j;
                            end
                            ROIimg = IMG.*uint8(BW);
                        elseif cropIMGon == 1 
                            if ROIshape_ind == 1   % use cropped ROI image
                                ROIcoords=separate_rois.(ROInames{k}).roi;
                                a=round(ROIcoords(1));b=round(ROIcoords(2));c=round(ROIcoords(3));d=round(ROIcoords(4));
                                ROIimg = IMG(b:b+d-1,a:a+c-1); % YL to be confirmed
                                % add boundary conditions
                                if ~isempty(BWcell)
                                    ROIbw  =  BWcell(b:b+d-1,a:a+c-1);
                                else
                                    ROIbw = [];
                                end
                                xc = round(a+c/2); yc = round(b+d/2);
                                disp('Cropped ROI only works with retanglar shape')
                            else
                                error('Cropped image ROI analysis for shapes other than rectangle is not availabe so far')
                            end
                        end
                       roiNamelist = ROInames{k};  % roi name on the list
                       if numSections > 1
                           roiNamefull = [fileNameNE,sprintf('_s%d_',j),roiNamelist,'.tif'];
                       elseif numSections == 1
                           roiNamefull = [fileNameNE,'_',roiNamelist,'.tif'];
                       end
                       if postFLAG == 0
                           imwrite(ROIimg,fullfile(ROIimgDir,roiNamefull));
                           %add ROI .tiff boundary name
                           if ~isempty(BWcell)
                               roiBWname = sprintf('mask for %s.tif',roiNamefull);
                               if ~exist(fullfile(ROIimgDir,'CA_Boundary'),'dir')
                                   mkdir(fullfile(ROIimgDir,'CA_Boundary'));
                               end
                               imwrite(ROIbw,fullfile(ROIimgDir,'CA_Boundary',roiBWname));
                               ROIbdryImg = ROIbw;
                               ROIcoords =  bwboundaries(ROIbw,4);
                           else
                               ROIbdryImg = [];
                               ROIcoords =  [];
                           end
                           [~,roiNamefullNE] = fileparts(roiNamefull);
                           try
                               [~,stats] = processROI(ROIimg, roiNamefullNE, ROIanaBatOutDir, keep, ROIcoords, distThresh, makeAssocFlag, makeMapFlag, makeOverFlag, makeFeatFlag, 1,infoLabel, bndryMode, ROIbdryImg, ROIimgDir, fibMode, advancedOPT,1);
                               ANG_value = stats(1);  % orientation
                               ALI_value = stats(5);  % alignment
                               % count the number of features from the output feature file
                               feaFilename = fullfile(ROIanaBatOutDir,[roiNamefullNE '_fibFeatures.csv']);
                               if exist(feaFilename,'file')
                                   fibNUM = size(importdata(feaFilename),1);
                               else
                                   fibNUM = nan;
                               end
                           catch EXP1
                               ANG_value = nan; ALI_value = nan;
                               fibNUM = nan;
                               disp(sprintf('%s was skipped in batchc-mode ROI analysis. Error message: %s',roiNamefull,EXP1.message))
                           end
                           if numSections > 1
                               z = j;
                           else
                               z = 1;
                           end
                           if cropIMGon == 1
                               cropFLAG = 'YES';   % analysis based on cropped image
                           elseif cropIMGon == 0
                               cropFLAG = 'NO';    % analysis based on orignal image with the region other than the ROI set to 0.
                           end
                           postFLAGt = 'NO'; % Yes: use post-processing based on available results in the output folder
                           if fibMode == 0 % "curvelets"
                               modeID = 'Curvelets'; 
                           else %"CTF fibers" 1,2,3
                               modeID = 'CTF Fibers';
                           end
                           if bndryMode == 0
                               bndryID = 'NO';
                           elseif bndryMode == 2 || bndryMode == 3
                               bndryID = 'YES';
                           end
                           CA_data_add = {items_number_current,sprintf('%s',fileNameNE),...
                           sprintf('%s',roiNamelist),sprintf('%.1f',ANG_value),sprintf('%.2f',ALI_value),...
                           sprintf('%d',fibNUM),modeID,bndryID,cropFLAG,postFLAGt,ROIshapes{ROIshape_ind},xc,yc,z};
%                            CA_data_add = {items_number_current,sprintf('%s',fileNameNE),sprintf('%s',roiNamelist),ROIshapes{ROIshape_ind},xc,yc,z,stats(1),stats(5)};
                           CA_data_current = [CA_data_current;CA_data_add];
                           set(CA_output_table,'Data',CA_data_current)
                           set(CA_table_fig,'Visible', 'on'); figure(CA_table_fig)
                       elseif postFLAG == 1
                           ROIfeasFLAG = 0;
                           try
                               %plot ROI k
                               B=bwboundaries(BW);
                               figure(guiFig);
                               for k2 = 1:length(B)
                                   boundary = B{k2};
                                   plot(boundary(:,2), boundary(:,1), 'm', 'LineWidth', 1.5);%boundary need not be dilated now because we are using plot function now
                               end
                               text(xc-10, yc,sprintf('%s',roiNamelist),'fontsize',5,'color','m','parent',imgAx)
                               clear B k2
                               
                               fiber_data = [];  % clear fiber_data
                               for ii = 1: size(fibFeat_load,1)
                                   ca = fibFeat_load(ii,4)*pi/180;
                                   xcf = fibFeat_load(ii,3);
                                   ycf = fibFeat_load(ii,2);
                                   if bndryMode == 0
                                       if BW(ycf,xcf) == 1
                                           fiber_data(ii,1) = k;
                                       elseif BW(ycf,xcf) == 0;
                                           fiber_data(ii,1) = 0;
                                       end
                                   elseif bndryMode >= 1   % boundary conditions
                                       % only count fibers/cuvelets that are within the
                                       % specified distance from the boundary  and within the
                                       % ROI defined here while excluding those within the tumor
                                       fiber_data(ii,1) = 0;
                                       % within the outside boundary distance but not within the inside
                                       ind2 = find((fibFeat_load(:,28) <= distThresh & fibFeat_load(:,29) == 0) == 1); 
                                       if ~isempty(find(ind2 == ii))
                                           if BW(ycf,xcf) == 1
                                               fiber_data(ii,1) = k;
                                           end
                                       end
                                   end   %bndryMode
                               end  %ii: length of fiber features
                               if bndryMode == 0
                                   featureLABEL = 4;
                                   featurename = 'Absolute Angle';
                               elseif bndryMode >= 1
                                   featureLABEL = 30 ;
                                   featurename = 'Relative Angle';
                               end
                               if numSections == 1
                                   csvFEAname = [fileNameNE '_' roiNamelist '_fibFeatures.csv']; % csv name for ROI k
                                   matFEAname = [fileNameNE '_' roiNamelist '_fibFeatures.mat']; % mat name for ROI k
                                   ROIimgname =  [fileNameNE '_' roiNamelist];
                               elseif numSections > 1
                                   csvFEAname = [fileNameNE sprintf('_s%d_',j) roiNamelist '_fibFeatures.csv']; % csv name for ROI k
                                   matFEAname = [fileNameNE sprintf('_s%d_',j) roiNamelist '_fibFeatures.mat']; % mat name for ROI k
                                   ROIimgname =  [fileNameNE sprintf('_s%d_',j) roiNamelist];
                               end
                               ind = find( fiber_data(:,1) == k);
                               fibFeat = fibFeat_load(ind,:);
                               fibNUM = size(fibFeat,1);
                               % save data of the ROI
                               csvwrite(fullfile(ROIpostBatDir,csvFEAname), fibFeat);
                               disp(sprintf('%s  is saved', fullfile(ROIpostBatDir,csvFEAname)))
                               matdata_CApost.fibFeat = fibFeat;
                               save(fullfile(ROIpostBatDir,matFEAname), 'fibFeat','tifBoundary','fibProcMeth','distThresh','coords');
                               % statistical analysis on the ROI features;
                               ROIfeature = fibFeat(:,featureLABEL);
                           catch
                               ROIfeasFLAG = 1;fibNUM = nan;
                               disp(sprintf('%s, ROI %d  ROI feature files is skipped',IMGname,k))
                           end
                          ROIstatsFLAG = 0;
                          try
                              stats = makeStatsOROI(ROIfeature,ROIpostBatDir,ROIimgname,bndryMode);
                              ANG_value = stats(1);  % orientation
                              ALI_value = stats(5);  % alignment
                          catch EXP2
                              ANG_value = nan; ALI_value = nan;
                              ROIstatsFLAG = 1;
                              disp(sprintf('%s, ROI %d  ROI stats is skipped. Error message:%s',IMGname,k,EXP2.message))
                          end
                           if numSections > 1
                               z = j;
                           else
                               z = 1;
                           end
                           CA_data_add = {items_number_current,sprintf('%s',fileNameNE),...
                               sprintf('%s',roiNamelist),sprintf('%.1f',ANG_value),sprintf('%.2f',ALI_value),...
                               sprintf('%d',fibNUM),modeID,bndryID,cropFLAG,postFLAGt,ROIshapes{ROIshape_ind},xc,yc,z};
                           CA_data_current = [CA_data_current;CA_data_add];
                           set(CA_output_table,'Data',CA_data_current)
                           set(CA_table_fig,'Visible', 'on'); figure(CA_table_fig)
                       end %postFLAG
               end % k: ROI number
               hold off % guiFig
               % save overlaid image with ROIname
               if postFLAG == 1   % post-processing of the CA features
                   if numSections  == 1
                       saveOverlayROIname = fullfile(ROIpostBatDir,[fileNameNE,'_overlay_ROIs.tif']);
                   else
                       saveOverlayROIname = fullfile(ROIpostBatDir,sprintf('%s_s%d_overlay_ROIs.tif',fileNameNE,j));
                   end
                   set(guiFig,'PaperUnits','inches','PaperPosition',[0 0 size(img,2)/200 size(img,1)/200]);
                   print(guiFig,'-dtiffn', '-r200', saveOverlayROIname);%YL, '-append'); %save a temporary copy of the image
               end
           end % j: slice number
       end %i: file number
   if ~isempty(CA_data_current)
           save(fullfile(ROImanDir,'last_ROIsCA.mat'),'CA_data_current','separate_rois')
           if postFLAG == 1
               existFILE = length(dir(fullfile(ROIpostBatDir,'Batch_ROIsCApost*.xlsx')));
               try
                   xlswrite(fullfile(ROIpostBatDir,sprintf('Batch_ROIsCApost%d.xlsx',existFILE+1)),...
                       [columnname;CA_data_current],'CA ROI alignment analysis') ;
               catch
                   xlwrite(fullfile(ROIpostBatDir,sprintf('Batch_ROIsCApost%d.xlsx',existFILE+1)),...
                       [columnname;CA_data_current],'CA ROI alignment analysis') ;
               end
               info_temp = 'Click the item(s) in the output table to check the tracked fibers in each ROI.' ;
               set(infoLabel,'String',sprintf('Done with the CA ROI analysis, results were saved into %s \n %s',...
                   fullfile(ROIpostBatDir,sprintf('Batch_ROIsCApost%d.xlsx',existFILE+1)),info_temp))
           elseif postFLAG == 0
               existFILE = length(dir(fullfile(ROIimgDir,'Batch_ROIsCAana*.xlsx')));
               try
                   xlswrite(fullfile(ROIimgDir,sprintf('Batch_ROIsCAana%d.xlsx',existFILE+1)),...
                       [columnname;CA_data_current],'CA ROI alignment analysis') ;
               catch
                   xlwrite(fullfile(ROIimgDir,sprintf('Batch_ROIsCAana%d.xlsx',existFILE+1)),...
                       [columnname;CA_data_current],'CA ROI alignment analysis') ;
               end
               info_temp = 'Click the item(s) in the output table to check the tracked fibers in each ROI.' ;
               set(infoLabel,'String',sprintf('Done with the CA post_ROI analysis, results were saved into %s.\n %s',...
                   fullfile(ROIimgDir,sprintf('Batch_ROIsCAana%d.xlsx',existFILE+1)),info_temp))
           end
   end
   disp('Done!') 
%    %clean up the displayed 
%    CA_OLfig_h = findobj(0,'Name','CurveAlign Fiber Overlay');
%    CA_MAPfig_h = findobj(0,'Name','CurveAlign Angle Map');
%    if ~isempty(CA_OLfig_h)
%        close(CA_OLfig_h)
%        disp('The CurveAlign overlay figure is closed')
%    end
%    if ~isempty(CA_MAPfig_h)
%        close(CA_MAPfig_h)
%        disp('The CurveAlign Angle heatmap is closed')
%    end
   disp('Click the item(s) in the output table to check the tracked fibers in each ROI.')
   figure(CA_table_fig)
   end
%%--------------------------------------------------------------------------
%%callback function for CAFEApost button
    function CAFEApost_Callback(CAFEApost,evendata)
        set(bndryModeDrop,'Enable','off');
        set(fibModeDrop,'Enable','off');
        set(imgOpen,'Enable','off');
        set(infoLabel,'String','Select the CA_Out folder and the features to be outputed');
        % add the options to postprocessing the output file
        set(CApostgcf,'Visible','on');
        return
    end
%--------------------------------------------------------------------------
% callback function for CApostfolderopen
    function CApostfolderopen_Callback(hObject,eventdata)
       
       CApostfolder = uigetdir(CApostfolder,'Selected CA output folder');
       if  CApostfolder == 0
           disp('No CA output folder is selected for post-processing.')
           CApostfolder =  CApostOptions.CApostfilepath;
           return
       else
           CApostOptions.CApostfilepath = CApostfolder;
           set(CApostfolderinfo,'String',CApostfolder);
       end
    end
%--------------------------------------------------------------------------
% callback function for HE_threshold_eidt_Callback text box
    function CApostgcfOK_Callback(hObject,eventdata)
        
        if isempty(CApostOptions.CApostfilepath) 
            disp('CA output folder is not selected for post-processing.')
            return
        end
        CApostOptions.RawdataFLAG = (get(combine_featurefiles,'Value') == get(combine_featurefiles,'Max'));
        CApostOptions.ALLstatsFLAG = (get(makeCAstats_all,'Value') == get(makeCAstats_all,'Max'));
        CApostOptions.SELstatsFLAG = (get(makeCAstats_exist,'Value') == get(makeCAstats_exist,'Max'));
        if CApostOptions.RawdataFLAG == 0 && CApostOptions.ALLstatsFLAG == 0 && CApostOptions.SELstatsFLAG== 0
            disp('At least one box is needed to be checked for a CA feature post-processing')
            return
        end
        CApostOptionsTEMP = CApostOptions;
        % select the folder where the CA out put is saved
        fibFeatDir = CApostOptions.CApostfilepath
        pathNameGlobal = fibFeatDir;
        save('currentP_CA.mat','pathNameGlobal','keepValGlobal','distValGlobal');
        % list feature names and output options
        fileList = dir(fullfile(fibFeatDir,'*fibFeatures*.csv'));
        if isempty(fileList)
            error('Feature files not exist')
        end
        lenFileList = length(fileList);
        feat_idx = zeros(1,lenFileList);
        %Search for feature files
        alignmentfiles = 0;
        compFeat = nan(lenFileList,36);
        OUTfiles = {};
        OUTcombined = cell(lenFileList,38); % first column: label, second: image name
        % set filenme for combined feature files
        if CApostOptions.RawdataFLAG == 1
           Combined_RAW_list = dir(fullfile(CApostOptions.CApostfilepath,'combined_RAWfeaturefiles*.xlsx'));
           if ~isempty(Combined_RAW_list)
              Combined_RAW_name = sprintf('combined_RAWfeaturefiles%d.xlsx', length(Combined_RAW_list)+1);
           else
              Combined_RAW_name = 'combined_RAWfeaturefiles1.xlsx';
           end
           FEAraw_combined_filename = fullfile(CApostOptions.CApostfilepath, Combined_RAW_name);  
        end
        % set filenme for the combined statistics for all features
         if CApostOptions.ALLstatsFLAG == 1
           Combined_STA_ALLlist = dir(fullfile(CApostOptions.CApostfilepath,'Combined_statistics_ALLfibFeatures*.xlsx'));
           if ~isempty(Combined_STA_ALLlist)
              Combined_STA_ALLname = sprintf('Combined_statistics_ALLfibFeatures%d.xlsx', length(Combined_STA_ALLlist)+1);
           else
              Combined_STA_ALLname = 'Combined_statistics_ALLfibFeatures1.xlsx';
           end
           CAOUTcombinedSTAname_ALL = fullfile(CApostOptions.CApostfilepath, Combined_STA_ALLname);  
        end
          % set filenme for the combined statistics for selected features
         if CApostOptions.SELstatsFLAG == 1
           Combined_STA_SELlist = dir(fullfile(CApostOptions.CApostfilepath,'Combined_statistics_SELfibFeatures*.xlsx'));
           if ~isempty(Combined_STA_SELlist)
              Combined_STA_SELname = sprintf('Combined_statistics_SELfibFeatures%d.xlsx', length(Combined_STA_SELlist)+1);
           else
              Combined_STA_SELname = 'Combined_statistics_SELfibFeatures1.xlsx';
           end
           CAOUTcombinedSTAname_SEL = fullfile(CApostOptions.CApostfilepath, Combined_STA_SELname);  
        end
        
        for i = 1:lenFileList
            fea_data =  importdata(fullfile(fibFeatDir,fileList(i).name));
            if size(fea_data,2)< 34
                tempFEA = nan(size(fea_data,1),34);
                tempFEA(1:size(fea_data,1),1:size(fea_data,2)) = fea_data;
                fea_data = tempFEA;
            end
            if CApostOptions.RawdataFLAG == 1
                if i == 1
                    FEAraw_combined = fea_data;
                else
                    FEAraw_combined = vertcat(FEAraw_combined,fea_data);
                end
            end
            compFeat(i,1:size(fea_data,2)) = nanmean(fea_data);
            
            if ~isempty(findstr(fileList(i).name,'_fibFeatures.csv'))
                filenameNE = strrep(fileList(i).name,'_fibFeatures.csv','');
                OUTfiles = [OUTfiles;{filenameNE}];
                filenameALI = fullfile(fibFeatDir,[filenameNE '_stats.csv']);
            else
                strstart = findstr(fileList(i).name,'_fibFeatures');
                filenametemp = strrep(fileList(i).name,'_fibFeatures','');
                [~,filenameNE] = fileparts(filenametemp);
                OUTfiles = [OUTfiles;{filenameNE}];
                filenameALI = fullfile(fibFeatDir,[filenameNE '_stats.csv']);
                
            end
            disp(sprintf('Searching for overall alignment files, %d of %d', i,lenFileList));
            if ~exist(filenameALI,'file')
                disp(sprintf('%s not exist, overall alignment not exist', filenameALI))
            else
                disp(sprintf('%s exists, overall alignment will be output', filenameALI))
                alignmentfiles = alignmentfiles + 1;
                statsOUT = importdata(filenameALI,'\t');
                try
                    compFeat(i,35) =  str2num(strrep(statsOUT{1},'Mean','')); % primary orientation
                    compFeat(i,36) =  str2num(strrep(statsOUT{5},'Coef of Alignment','')); % alignment coefficient
                    compFeat(i,37) =  str2num(strrep(statsOUT{2},'Median','')); %
                    compFeat(i,38) =  str2num(strrep(statsOUT{3},'Variance','')); %
                    compFeat(i,39) =  str2num(strrep(statsOUT{4},'Std Dev','')); %
                    compFeat(i,40) =  str2num(strrep(statsOUT{6},'Skewness','')); %
                    compFeat(i,41) =  str2num(strrep(statsOUT{7},'Kurtosis','')); %
                    compFeat(i,42) =  str2num(strrep(statsOUT{8},'Omni Test','')); %
                    compFeat(i,43) =  str2num(strrep(statsOUT{9},'red pixels','')); %
                    compFeat(i,44) =  str2num(strrep(statsOUT{10},'yellow pixels','')); %
                    compFeat(i,45) =  str2num(strrep(statsOUT{11},'green pixels','')); %
                    % resolve compatibility with older versions: add total pixels as an output
                    try
                        compFeat(i,46) =  str2num(strrep(statsOUT{11},'total pixels','')); %
                    catch
                        compFeat(i,46) = nan;
                    end
                catch
                    compFeat(i,35) =  statsOUT.data(1); % primary orientation
                    compFeat(i,36) =  statsOUT.data(5); % alignment coefficient
                    compFeat(i,37) =  statsOUT.data(2); %
                    compFeat(i,38) =  statsOUT.data(3); %
                    compFeat(i,39) =  statsOUT.data(4); %
                    compFeat(i,40) =  statsOUT.data(6); %
                    compFeat(i,41) =  statsOUT.data(7); %
                    compFeat(i,42) =  statsOUT.data(8); %
                    compFeat(i,43) =  statsOUT.data(9); %
                    compFeat(i,44) =  statsOUT.data(10);%
                    compFeat(i,45) =  statsOUT.data(11);%
                    % add total pixels as an output
                    try
                        compFeat(i,46) =  statsOUT.data(12);%
                    catch
                        compFeat(i,46) = nan;
                    end
                end
            end
            OUTcombined{i,1} = i;
            OUTcombined{i,2} = filenameNE;
            OUTcombined(i,3:size(compFeat,2)+2) = num2cell(compFeat(i,:));
        end
        disp(sprintf('found %d alignment files from %d files', alignmentfiles,lenFileList))
        
        featNames = {...
            'fiber Key into CTFIRE list', ...
            'end point row', ...
            'end point col', ...
            'fiber abs ang', ...
            'fiber weight', ...
            'total length', ...
            'end to end length', ...
            'curvature', ...
            'width', ...
            'dist to nearest MNF(minimum nearest fibers)', ...
            'dist to nearest 2MNF', ...
            'dist to nearest 4MNF', ...
            'dist to nearest 8MNF', ...
            'mean nearest dist', ...
            'std nearest dist', ...
            'box density MBS(minimum box size)', ...
            'box density 2MBS', ...
            'box density 4MBS', ...
            'alignment of nearest MNF(minimum nearest fibers)', ...
            'alignment of nearest 2MNF', ...
            'alignment of nearest 4MNF', ...
            'alignment of nearest 8MNF', ...
            'mean nearest align', ...
            'std nearest align', ...
            'box alignment MBS(minimum box size)', ...
            'box alignment 2MBS', ...
            'box alignment 4MBS', ...
            'nearest dist to bound', ...
            'inside epi region', ...
            'nearest relative boundary angle', ...
            'extension point distance', ...
            'extension point angle', ...
            'boundary point row', ...
            'boundary point col'};
        
        %1. fiber Key into CTFIRE list
        %2. row
        %3. col
        %4. abs ang
        %5. fiber weight
        %6. total length
        %7. end to end length
        %8. curvature
        %9. width
        %10. dist to nearest 2
        %11. dist to nearest 4
        %12. dist to nearest 8
        %13. dist to nearest 16
        %14. mean dist (8-11)
        %15. std dist (8-11)
        %16. box density 32
        %17. box density 64
        %18. box density 128
        %19. alignment of nearest 2
        %20. alignment of nearest 4
        %21. alignment of nearest 8
        %22. alignment of nearest 16
        %23. mean align (14-17)
        %24. std align (14-17)
        %25. box alignment 32
        %26. box alignment 64
        %27. box alignment 128
        %28. nearest dist to bound
        %29. nearest dist to region
        %30. nearest relative boundary angle
        %31. extension point distance
        %32. extension point angle
        %33. boundary point row
        %34. boundary point col
        %Save fiber feature array
         if CApostOptions.RawdataFLAG == 1
             try
                 xlswrite(FEAraw_combined_filename,featNames,'featureData_combined','A1');
                 xlswrite(FEAraw_combined_filename,FEAraw_combined,'featureData_combined','A2');
                 xlswrite(FEAraw_combined_filename,(extractfield(fileList,'name'))','files_combined');
             catch
                 xlwrite(FEAraw_combined_filename,featNames,'featureData_combined','A1');
                 xlwrite(FEAraw_combined_filename,FEAraw_combined,'featureData_combined','A2');
                 xlwrite(FEAraw_combined_filename,(extractfield(fileList,'name'))','files_combined');
             end
           %% add a 'statistics' sheet to save the statistics of the combine raw data
           statsName_raw = {'Median','Mode','Mean','Variance','Std','Min','Max','CountedFibers','Skewness','Kurtosis'}';%statistical measures include
           stats_raw(1,:) = nanmedian(FEAraw_combined);
           stats_raw(2,:) = mode(FEAraw_combined);
           stats_raw(3,:) = nanmean(FEAraw_combined);
           stats_raw(4,:) = nanvar(FEAraw_combined);
           stats_raw(5,:) = nanstd(FEAraw_combined);
           stats_raw(6,:) = min(FEAraw_combined);
           stats_raw(7,:) = max(FEAraw_combined);
           for j = 1:size(FEAraw_combined,2)
               stats_raw(8,j) = length(find(~isnan(FEAraw_combined(:,j))== 1));  % count ~nan number
           end
           stats_raw(9,:) = skewness(FEAraw_combined); %measure of symmetry
           stats_raw(10,:) = kurtosis(FEAraw_combined); %measure of peakedness
           stats_raw(:,[1 5 29]) = nan;  % set'fiber Key into CTFIRE list', 'fiber weight' and 'inside epi region' set to nan for a statistical analysis
           try
               xlswrite(FEAraw_combined_filename,featNames,'statistics','B1');
               xlswrite(FEAraw_combined_filename, stats_raw,'statistics','B2');
               xlswrite(FEAraw_combined_filename,statsName_raw,'statistics','A2');
           catch
               xlwrite(FEAraw_combined_filename,featNames,'statistics','B1');
               xlwrite(FEAraw_combined_filename, stats_raw,'statistics','B2');
               xlwrite(FEAraw_combined_filename,statsName_raw,'statistics','A2');
           end
           disp(sprintf('Combined feature files is saved in %s',FEAraw_combined_filename)) ;
         end
   
        aliNames = {'overall orientation','overall alignment','angle median',...
            'angle variance','angle std','angle skewness','angle Kurtosis',...
            'Omni Test','red pixels','yellow pixels','green pixels','total pixels'};   % alignment
        outNamesall = [featNames,aliNames];
        
        Nnanflag = ~isnan(compFeat(1,:));
        outNamesall_index = find(Nnanflag== 1);
        outNames_Selected = outNamesall(outNamesall_index);
        compFeatOUT = compFeat(outNamesall_index);
        columnnameCOM = [{'No.'},{'image label'},outNames_Selected];
        columnnameALL = [{'No.'},{'image label'},outNamesall];
        CAdata_combined =  OUTcombined(:,[1 2 outNamesall_index+2]);
        
        
        if CApostOptions.ALLstatsFLAG == 1
            try
                xlswrite(CAOUTcombinedSTAname_ALL,columnnameALL,'CAcombined','A1');
                xlswrite(CAOUTcombinedSTAname_ALL,OUTcombined,'CAcombined','A2');
            catch
                xlwrite(CAOUTcombinedSTAname_ALL,columnnameALL,'CAcombined','A1');
                xlwrite(CAOUTcombinedSTAname_ALL,OUTcombined,'CAcombined','A2');
            end
            disp(sprintf('Combined average value for all features is saved in %s',CAOUTcombinedSTAname_ALL));
            
        end
        
        if CApostOptions.SELstatsFLAG == 1
            try
                xlswrite(CAOUTcombinedSTAname_SEL,columnnameCOM,'CAcombined','A1');
                xlswrite(CAOUTcombinedSTAname_SEL,CAdata_combined,'CAcombined','A2');
            catch
                xlwrite(CAOUTcombinedSTAname_SEL,columnnameCOM,'CAcombined','A1');
                xlwrite(CAOUTcombinedSTAname_SEL,CAdata_combined,'CAcombined','A2');
            end
            disp(sprintf('Combined average value for selected features is saved in %s',CAOUTcombinedSTAname_SEL));
            
        end
        %output table for selected features
        % Column names and column format
        columnnameSEL = columnnameCOM;
        columnformatSEL = [{'numeric'},{'char'},repmat({'numeric'},1,length(columnnameCOM)-2)];
        % Create the uitable
        CAsel_table_fig = figure(243);clf
        %      figPOS = get(caIMG_fig,'Position');
        %      figPOS = [figPOS(1)+0.5*figPOS(3) figPOS(2)+0.75*figPOS(4) figPOS(3)*1.25 figPOS(4)*0.275]
        figPOSsel = [0.2 0.45 0.725 0.425];
        set(CAsel_table_fig,'Units','normalized','Position',figPOSsel,'Visible','on','NumberTitle','off')
        set(CAsel_table_fig,'name',sprintf('CurveAlign combined output table'))
        CAsel_output_table = uitable('Parent',CAsel_table_fig,'Units','normalized','Position',[0.05 0.05 0.9 0.9],...
            'Data', CAdata_combined,...
            'ColumnName', columnnameSEL,...
            'RowName',[],...
            'ColumnFormat',columnformatSEL);
    end

%-----------------------------------------------------------------------
% callback function for HE_threshold_eidt_Callback text box
    function CApostgcfCANCEL_Callback(hObject,eventdata)
       
        set(CApostgcf,'Visible', 'off')
        disp('CA post-processing is cancelled ')
  
    end
        
%--------------------------------------------------------------------------
%%--------------------------------------------------------------------------
%callback function for feature ranking button
function featR(featRanking,eventdata)
    
   if(exist(fullfile(pathNameGlobal,'annotation.xlsx'),'file')==2)
      %annotation file exists - open a dialogue box to ask the user if new
      %annotation file must be formed or not
      choice=questdlg('annotation file already present.Do you wish to create a new one or use the current one ?','Question','Create New one','Use current one','Cancel','Cancel');
      if(isempty(choice))
         return; 
      else
          switch choice
              case 'Create New one'
                    feature_ranking_automation_fn;
              case 'Use current one'
                    
              case 'Cancel'
                  return;
          end
              
      end
   else
      %annotation file does not exist
      feature_ranking_automation_fn;
   end
    set(bndryModeDrop,'Enable','off');
    set(fibModeDrop,'Enable','off');
    set(imgOpen,'Enable','off');
    set(infoLabel,'String','Feature Ranking is ongoing');

%     fibFeatDir=pathNameGlobal;
    if(isempty(pathNameGlobal)==0)
       fibFeatDir=fullfile(pathNameGlobal); 
    else
        fibFeatDir = uigetdir(pathNameGlobal,'Select Fiber feature Directory:');
        pathNameGlobal=fibFeatDir(1:strfind(fibFeatDir,'CA_Out')-1);
    end
    
     
%      pathNameGlobal = fibFeatDir;
     save('currentP_CA.mat','pathNameGlobal','keepValGlobal','distValGlobal');
    fileList = dir(fibFeatDir);
    lenFileList = length(fileList);
    feat_idx = zeros(1,lenFileList);
    %Search for feature files
    for i = 1:lenFileList
       disp(sprintf('Searching for feature files, %d of %d', i,lenFileList));
        if ~isempty(regexp(fileList(i).name,'fibFeatures.mat', 'once', 'ignorecase'))
            feat_idx(i) = 1;
        end
    end
    featFiles = fileList(feat_idx==1);
    lenFeatFiles = length(featFiles);
    %Compile a big array of features in RAM
    obsFileIdx = zeros(lenFeatFiles,1);
    totFeat = 0;
    for i = 1:lenFeatFiles
        %Find how many observations are in each file
        obsName = featFiles(i).name;
        bff = [fibFeatDir obsName];
        feat = load(bff);
        [lenFeat widFeat] = size(feat.fibFeat);
        disp(sprintf('%d, there are %d features in %s, ',i,widFeat,obsName)); % YL for debug
        totFeat = totFeat + lenFeat;
        obsFileIdx(i) = totFeat;
%         IMGname1{i,1} = obsName(1:end-16);  % original image name of the feature name
        IMGname1{i,1} = strrep(obsName,'_fibFeatures.mat','');  % original image name without the extension
        disp(sprintf('Counting observations in each file, %d of %d', i,lenFeatFiles));

    end
    
    %Allocate space for complete feature array (For all images)
    compFeat = zeros(totFeat,widFeat+1);
    compFeatMeta(lenFeatFiles) = struct('imageName',[],'topLevelDir',[],'fireDir',[],'outDir',[],'numToProc',[],'fibProcMeth',[],'keep',[],'distThresh',[]);

    %Put together big array
    prevTot = 1;
    totFeat = 0;
    for i = 1:lenFeatFiles
        obsName = featFiles(i).name;
        bff = [fibFeatDir obsName];
        feat = load(bff);
        disp(sprintf('loading feature data, %d of %d', i,lenFeatFiles));
        [lenFeat widFeat] = size(feat.fibFeat);
        totFeat = totFeat + lenFeat; %Pointer to last array position
        compFeat(prevTot:totFeat,1:end-1) = feat.fibFeat; %Add to array
        compFeat(prevTot:totFeat,end) = zeros(lenFeat,1)+i; %Add index into meta data array
        compFeatMeta(i).imageName = feat.imgNameP;
        prevTot = totFeat+1; %Pointer to first array position
    end
    featNames = feat.featNames;

    %Save feature array and meta array to disk
    compFeatFF = [fibFeatDir 'compFeat.mat'];
    save(compFeatFF,'compFeat','compFeatMeta');
  [labelMeta2 IMGname2 rawANN] = xlsread([fibFeatDir,'annotation.xlsx']);
  for i = 1:length(IMGname1)
      for j = 1:length(IMGname2)
          if strcmp(IMGname1(i),IMGname2(j))
             labelMeta(1,i) = labelMeta2(j,1);
              break;
          end
      end
  end
    [lenFeat widFeat] = size(compFeat); %get size of the complete feature matrix (including meta index)
    % find the not 'Nan' features
    Nnanflag = ~isnan(compFeat(1,:));
    Nnanfeat = find(Nnanflag(1:end-1) == 1);  % find the features whos value is not a 'NaN' value, throw out the last col which contains index to metaData; 
    labelObs = zeros(lenFeat,1); %The label for each observation
    for i = 1:lenFeat
        labelObs(i) = labelMeta(compFeat(i,end)); %last col contains index to metaData
    end
    labelObs = logical(labelObs);
    %show featurelist
    guiRank1 = findobj(0,'Name','CA Features List');
    if isempty(guiRank1)
        guiRank1 = figure('Resize','on','Units','normalized','Position',...
            [0.265 0.25 0.30 0.60],'Visible','off','MenuBar','none',...
            'name','CA Features List','NumberTitle','off','UserData',0);
    end
    figure(guiRank1);
    
    fzz1 = 9;
    for i = 1:length(featNames)
        if i<13
            text(0,1-0.08*i,sprintf('%d: %s',i,featNames{i}),'fontsize',fzz1);
        elseif i > 12 && i < 25
            text(0.34,1-0.08*(i-12),sprintf('%d: %s',i,featNames{i}),'fontsize',fzz1);
        elseif i > 24
            text(0.68,1-0.08*(i-24),sprintf('%d: %s',i,featNames{i}),'fontsize',fzz1);
        end
    end
    axis(gca,'off')
    posObs = compFeat(labelObs,:); %positive fiber observations
    negObs = compFeat(~labelObs,:); %negative fiber observations
    posM = nanmean(posObs); %average over observations
    negM = nanmean(negObs);
    posStd = nanstd(posObs);
    negStd = nanstd(negObs);
    compM = [posM; negM]; %composite matrix
    compStd = [posStd; negStd];
    maxM = nanmax(compM); %max between positive and neg
    compMN(1,:) = compM(1,:)./maxM; %normalize
    compMN(2,:) = compM(2,:)./maxM;
    compStdN(1,:) = compStd(1,:)./maxM;
    compStdN(2,:) = compStd(2,:)./maxM;
 
    featsDef = '10:27'; %Best feature set
    name = 'Select features to be ranked';
    pptinfo= sprintf('Select features among %s', strcat(num2str(Nnanfeat))); % show all the Not a NaN features
    % prompt= featNames';
    prompt = {pptinfo};
    numlines=1;
    defaultanswer= {featsDef};
    options.Resize='on';
    options.WindowStyle='normal';
    options.Interpreter='tex';
    featsIN = inputdlg(prompt,name,numlines,defaultanswer,options);

    if length(featsIN)== 1
        feats = str2num(featsIN{1});
    else
        disp('Please confirm or update the selected fiber features')
        featsIN = 0;
    end
    %feats = [28:32];
    featNamesS = featNames(feats); %throw out names that are not included
    lenSubFeats = length(feats);

    %Now make each image an observation by calculating the mean over fibers
    imgObsM = zeros(lenFeatFiles,widFeat);
    for i = 1:lenFeatFiles    
        imgObsM(i,:) = nanmean(compFeat(compFeat(:,end) == i,:));
    end
    %check to make sure these features can classify the training set
    SVMStruct = svmtrain(imgObsM(:,feats),labelMeta,'showplot','true');
    if lenSubFeats == 2
        xlabel(featNamesS(1));
        ylabel(featNamesS(2));
    end
    v = svmclassify(SVMStruct,imgObsM(:,feats));
    labelMeta = labelMeta';
    tp = sum((v(:,1) == labelMeta(:,1) & labelMeta(:,1) == 1));
    tn = sum((v(:,1) == labelMeta(:,1) & labelMeta(:,1) == 0));
    fp = sum((v(:,1) ~= labelMeta(:,1) & labelMeta(:,1) == 0));
    fn = sum((v(:,1) ~= labelMeta(:,1) & labelMeta(:,1) == 1));
    sens = tp/(tp+fn);
    spec = tn/(fp+tn);   
    disp(sprintf('mean sensitivity: %4.2f',mean(sens)));
    disp(sprintf('mean specificity: %4.2f',mean(spec)));

    %Plot normalized average feature values for each class
    guiRank2 = findobj(0,'Name','Feature Normalized Difference (Pos-Neg)');
    if isempty(guiRank2)
        guiRank2 = figure('Resize','on','Units','normalized','Position',...
            [0.575 0.53 0.30 0.42],'Visible','off','MenuBar','none',...
            'name','Feature Normalized Difference (Pos-Neg)','NumberTitle','off','UserData',0);
    end
    figure(guiRank2);
    %set(gcf,'Position',[1 1 1000 750]);
    difCompMN = compMN(1,:) - compMN(2,:);
    difCompStd = sqrt(compStdN(1,:).^2 + compStdN(2,:).^2); % YL: compStdN(2,:).^2?
    difMN = difCompMN(:,feats);
    difStdS = difCompStd(:,feats);
    [difS, idxS] = sort(difMN);
    barh(difS);
    featNamesS1 = featNamesS(idxS); %Sorted name list
    difStdS = difStdS(idxS)./10; %shrink to make plotable
    %xlim([0.2 1.0]);
    set(gca,'YTick',1:lenSubFeats,'YTickLabel',featNamesS1);
    set(gca,'XGrid','off','YGrid','on');
    xlabel('Normalized Difference (Pos-Neg)');
    for i = 1:lenSubFeats
        %plot relative error
        yp = lenSubFeats-i+1;
        line([difS(i)-difStdS(yp) difS(i)+difStdS(yp)],[i i],'Color','g');
    end
   
    text(min(difS)-max(difStdS),lenSubFeats+2.85,sprintf('Sensitivity: %4.2f; Specificity: %4.2f',mean(sens),mean(spec)),'color','r');
    %Plot feature rank
    wtSVM = SVMStruct.Alpha'*SVMStruct.SupportVectors;
    absWt = wtSVM.^2;
    [absWtS, idxS] = sort(absWt); %Sort based on importance

    guiRank3 = findobj(0,'Name','Feature Classification Importance');
    if isempty(guiRank3)
        guiRank3 = figure('Resize','on','Units','normalized','Position',...
            [0.575 0.02 0.30 0.42],'Visible','off','MenuBar','figure',...
            'name','Feature Classification Importance','NumberTitle','off','UserData',0);
    end
    figure(guiRank3); barh(absWtS); %plot bar graph
    featNamesS = featNamesS(idxS); %sort feature names
    set(gca,'YTick',1:lenSubFeats,'YTickLabel',featNamesS);
    xlabel('Classification Importance');
    text(min(absWtS),lenSubFeats+2.85,sprintf('Sensitivity: %4.2f; Specificity: %4.2f',mean(sens),mean(spec)),'color','r');

    %Save rank to file
%     fibFeatDir2 = pwd;
    featRankFF = [fibFeatDir 'featRank.txt'];
    fid = fopen(featRankFF,'w+');
    difMNS = difMN(idxS);
    for i = 1:lenSubFeats
        fprintf(fid,'%d\t%s\t%f\t%f\r\n',i,featNamesS{i},absWtS(i),difMNS(i));
    end   
    fclose(fid);

    %% Try to use each fiber as an observation
%     compFeat(isnan(compFeat)) = 0;
%     folds = 50;
%     rndIdx = logical(round(rand(lenFeat,1)*(0.5+1/folds)));
%     %train
%     SVMStruct = svmtrain([compFeat(rndIdx,feats)],labelObs(rndIdx),'kernel_function','linear','showplot','true');
%     if lenSubFeats == 2
%         xlabel(featNamesS(1));
%         ylabel(featNamesS(2));
%     end
%     v = svmclassify(SVMStruct,compFeat(:,feats));
%     tp = sum((v(:,1) == labelObs(:,1) & labelObs(:,1) == 1));
%     tn = sum((v(:,1) == labelObs(:,1) & labelObs(:,1) == 0));
%     fp = sum((v(:,1) ~= labelObs(:,1) & labelObs(:,1) == 0));
%     fn = sum((v(:,1) ~= labelObs(:,1) & labelObs(:,1) == 1));
%     sens = tp/(tp+fn);
%     spec = tn/(fp+tn);   
%     disp(sprintf('mean sensitivity: %f',mean(sens)));
%     disp(sprintf('mean specificity: %f',mean(spec)));
% 
%     wtSVM = SVMStruct.Alpha'*SVMStruct.SupportVectors;
%     absWt = wtSVM.^2;
%     figure(300); barh(absWt);

%      set([makeRecon makeAngle makeFeat makeOver makeMap imgRun],'Enable','on');
%      set([makeRecon makeAngle],'Enable','off') % yl,default output
   
  
%    set(fibModeDrop,'Enable','off');
   set(infoLabel,'String','Feature Ranking is done');

end  % featR
%--------------------------------------------------------------------------
% callback function for advanced options
    function advOptions_callback(handles, eventdata)
        
        name = 'Advanced Options';
        numlines = 0.9;
        optadv{1} = advancedOPT.exclude_fibers_inmaskFLAG;
        optadv{2} = advancedOPT.curvelets_group_radius;
        optadv{3} = advancedOPT.seleted_scale;
        optadv{4} = advancedOPT.heatmap_STDfilter_size;
        optadv{5} = advancedOPT.heatmap_SQUAREmaxfilter_size;
        optadv{6} = advancedOPT.heatmap_GAUSSIANdiscfilter_sigma;
        optadv{7} = advancedOPT.plotrgbFLAG;
        optadv{8} = advancedOPT.folderROIman;
        optadv{9} = advancedOPT.folderROIana;
        optadv{10} = advancedOPT.uniROIname;
        optadv{11} = advancedOPT.cropROI;
        optadv{12} = advancedOPT.specifyROIsize;
        optadv{13} = advancedOPT.minimum_nearest_fibers; mnf_adv =  optadv{13};
        optadv{14} = advancedOPT.minimum_box_size; mbs_adv =  optadv{14};
        optadv{15} = advancedOPT.fiber_midpointEST;
        optDefault= {num2str(optadv{1}), num2str(optadv{2}),num2str(optadv{3}),...
            num2str(optadv{4}),num2str(optadv{5}),num2str(optadv{6}),num2str(optadv{7}),...
            optadv{8},optadv{9},optadv{10},num2str(optadv{11}),num2str(optadv{12}),num2str(optadv{13}),num2str(optadv{14}),num2str(optadv{15})};
        promptname = {'Exclude fibers in tiff boundary flag,1: to exclude; 0: to keep',...
            'curvelets group radius [in pixels]','Scale to be used: 1: 2nd finest scale(default); 2: 3rd finest; and so on',...
            'Heatmap standard deviation filter for no-boundary case{in pixels)',...
            'Heatmap square max filter size(in pixels)',...
            'Heatmap Gaussian disc filter sigma( in pixels)',...
            'Flag for RGB image : 1: display RGB; 0: display grayscale',...
            'Folder for the ROI .mat files',...
            'Folder for the ROI analysis output',...
            'Unique part of the image name[set this if loading ROI file defined by another image]',...
            'Flag to crop and save rectangular ROI, 1: crop; 0: do not crop',...
            'Specify rectangular ROI size [width height]',...
            sprintf('Minimum nearest fibers (counted in feature list:%d,%d,%d,%d)',mnf_adv,2^1*mnf_adv,2^2*mnf_adv,2^3*mnf_adv),...
            sprintf('Minimum box size (counted in feature list :%d,%d,%d)',mbs_adv,2^1*mbs_adv,2^2*mbs_adv)...
            'Options for fiber middle point estimation based on: 1-end points coordinates(default);2-fiber length'};
        % FIREp = inputdlg(prompt,name,numlines,defaultanswer);
        optUpdate = inputdlg(promptname,name,numlines,optDefault);
        if isempty(optUpdate)
           disp('Advanced options are not changed')
           return
        end
        advancedOPT.exclude_fibers_inmaskFLAG = str2num(optUpdate{1});
        advancedOPT.curvelets_group_radius = str2num(optUpdate{2});
        advancedOPT.seleted_scale = str2num(optUpdate{3});
        advancedOPT.heatmap_STDfilter_size = str2num(optUpdate{4});
        advancedOPT.heatmap_SQUAREmaxfilter_size = str2num(optUpdate{5});
        advancedOPT.heatmap_GAUSSIANdiscfilter_sigma = str2num(optUpdate{6});
        advancedOPT.plotrgbFLAG = str2num(optUpdate{7});
        advancedOPT.folderROIman = optUpdate{8};
        advancedOPT.folderROIana = optUpdate{9};
        advancedOPT.uniROIname = optUpdate{10};
        advancedOPT.cropROI = str2num(optUpdate{11});
        advancedOPT.specifyROIsize = str2num(optUpdate{12});
        advancedOPT.minimum_nearest_fibers = str2num(optUpdate{13});
        advancedOPT.minimum_box_size = str2num(optUpdate{14});
        advancedOPT.fiber_midpointEST = str2num(optUpdate{15});
        %               try
        if strmatch(advancedOPT.folderROIman, '\\image path\ROI_management\','exact')
            advancedOPT.folderROIman = fullfile(pathName,'ROI_management');
            disp(sprintf('use the default ROI folder %s',advancedOPT.folderROIman))
            
        end
        
        
        if strmatch(advancedOPT.folderROIana, '\\image path\ROI_management\Cropped\','exact')
            advancedOPT.folderROIana = fullfile(pathName,'ROI_management','Cropped');
            disp(sprintf('use the default cropped image folder %s',advancedOPT.folderROIana))
        end
        
        if ~exist(advancedOPT.folderROIman)
            mkdir(advancedOPT.folderROIman)
        end
        
        if ~exist(advancedOPT.folderROIana)
            mkdir(advancedOPT.folderROIana)
        end
        
        if strmatch(advancedOPT.folderROIman, fullfile(pathName,'ROI_management'))
            loadROIFLAG = 0;
        else
            loadROIFLAG = 1;
        end
        
        if  advancedOPT.cropROI == 1
            
            set(infoLabel,'String',sprintf('Load ROI file from %s; \n Save cropped image in %s',...
                advancedOPT.folderROIman,advancedOPT.folderROIana))
        end
        % Map the ROI .mat file for each image to be cropped
        matfilelist = dir(fullfile(advancedOPT.folderROIman,'*ROIs.mat'));
        
        if length(fileName)> length(matfilelist)
            disp('one or more ROI files are missing')
        end
        
        for i = 1:length(matfilelist)
            matfileName{i} = matfilelist(i).name;
            
            if ~isempty(advancedOPT.uniROIname)
                matCommonName{i} = strrep( matfileName{i},'_ROIs.mat','');
                matCommonName{i} = strrep(matCommonName{i},advancedOPT.uniROIname,'');
            end
            
        end
        for i = 1:length(fileName)
            [~,IMGname,IMGext] = fileparts(fileName{i});
            if ~isempty(advancedOPT.uniROIname)
                IND1 = []; IND2 = [];
                for j = 1:length(matCommonName)
                    commonName = matCommonName{j};
                    N = length(commonName);
                    IND1(j) = strncmp(fileName{i},commonName,N);
                end
                
                IND2 = find(IND1 == 1);
                
                if isempty(IND2)
                    
                    error(sprintf('Could not find the ROI file for %s',fileName{i}));
                    
                elseif length(IND2) > 1
                    
                    error(sprintf('ROI file for %s is not unique',fileName{i}))
                elseif length(IND2) == 1
                    disp(sprintf('Found unique ROI file %s for image %s',matfileName{IND2}, fileName{i}))
                    
                end
                
                roiMATnamefull= matfileName{IND2};
            else
                
                roiMATnamefull = [IMGname,'_ROIs.mat'];
                if ~exist(fullfile(advancedOPT.folderROIman,roiMATnamefull))
                    disp(sprintf('%s not exist',roiMATnamefull))
                end
            end
           % crop the defined ROI(s)  
           if  advancedOPT.cropROI == 1
               
               if exist(fullfile(advancedOPT.folderROIman,roiMATnamefull),'file')
                   roiMATnameV{i} = roiMATnamefull;
                   load(fullfile(advancedOPT.folderROIman,roiMATnamefull),'separate_rois');
                   if isempty(separate_rois)
                       error(sprintf('No ROI defined in the %s',roiMATnamefull));
                   end
                   ROInames = fieldnames(separate_rois);
                   s_roi_num = length(ROInames);
                   
                   IMGnamefull = fullfile(pathName,fileName{i});
                   IMGinfo = imfinfo(IMGnamefull);
                   numSections = numel(IMGinfo); % number of sections
                   
                   if numSections == 1
                       
                       IMG = imread(IMGnamefull);
                       
                   elseif numSections > 1
                       
                       IMG = imread(IMGnamefull,1);
                       disp('only the first slice of the stack is loaded')
                       
                   end
                        
                   for k = 1:  s_roi_num
                       ROIshape_ind = separate_rois.(ROInames{k}).shape;
                       if ROIshape_ind == 1   % use cropped ROI image
                           ROIcoords=separate_rois.(ROInames{k}).roi;
                           a=ROIcoords(1);b=ROIcoords(2);c=ROIcoords(3);d=ROIcoords(4);
                          % add exception handling
                           if a< 1 || a+c-1> size(IMG,2)||b < 1 || b+d-1 > size(IMG,1)
                              disp(sprintf('%s of %s is out of range, and is skipped',ROInames{k},fileName{i}))
                               break 
                           end
                           ROIimg = [];
                           if size(IMG,3) == 1
                               ROIimg = IMG(b:b+d-1,a:a+c-1);
                           else
                               ROIimg = IMG(b:b+d-1,a:a+c-1,:);
                           end
                           
                           xc = round(a+c/2); yc = round(b+d/2);
                           imagename_crop = fullfile(advancedOPT.folderROIana,sprintf('%s_%s.tif',IMGname,ROInames{k}));
                           imwrite(ROIimg,imagename_crop);
                           %                                disp('cropped ROI was saved in ')
                           
                       else
                           error('Cropped image ROI analysis for shapes other than rectangle is not availabe so far')
                       end
                   end
                   if s_roi_num ==1
                       disp(sprintf('%d/%d, %d ROI was cropped', i,length(fileName),s_roi_num));
                   elseif  s_roi_num > 1
                       disp(sprintf('%d/%d, %d ROIs were cropped', i,length(fileName),s_roi_num))
                   end
                   
               end
           end %crop the defined ROI
                
        end  % i: fileName
    end
%--------------------------------------------------------------------------
% callback function for imgRun
    function runMeasure(imgRun,eventdata)
        
        %tempFolder = uigetdir(pathNameGlobal,'Select Output Directory:');
        outDir = fullfile(pathName, 'CA_Out');   
        outDir2 = fullfile(pathName, 'CA_Boundary');  
        if (~exist(outDir,'dir')||~exist(outDir2,'dir'))
            mkdir(outDir);mkdir(outDir2);
        end
                %         IMG = getappdata(imgOpen,'img');
        keep = get(enterKeep,'UserData');
        distThresh = get(enterDistThresh,'UserData');
        keepValGlobal = keep;
        distValGlobal = distThresh;
        save('currentP_CA.mat','pathNameGlobal','keepValGlobal','distValGlobal');
        
%         set([imgRun makeAngle makeRecon enterKeep enterDistThresh imgOpen makeAssoc makeFeat makeMap makeOver],'Enable','off')
        
        if isempty(keep)
            %indicates the % of curvelets to process (after sorting by
            %coefficient value)
            keep = .001;
        end
        
        if isempty(distThresh)
            %this is default and is in pixels
            distThresh = 100;
        end
        
        if bndryMode == 2 || bndryMode == 3
            setappdata(guiFig,'boundary',1)
        elseif bndryMode == 0
            coords = []; %no boundary
            bdryImg = [];
        else
              disp(sprintf('csv boundary file name: boundary for %s.csv',fileName{index_selected}))
        end
        %check if user directed to output boundary association lines (where
        %on the boundary the curvelet is being compared)
        makeAssocFlag = get(makeAssoc,'Value') == get(makeAssoc,'Max');
        
        makeFeatFlag = get(makeFeat,'Value') == get(makeFeat,'Max');
        makeOverFlag = get(makeOver,'Value') == get(makeOver,'Max');
        makeMapFlag = get(makeMap,'Value') == get(makeMap,'Max');
        %check to see if we should process the whole stack or current image
        %wholeStackFlag = get(wholeStack,'Value') == get(wholeStack,'Max');
        
        %loop through all images in batch list
        for k = 1:length(fileName)
            disp(['Processing image # ' num2str(k) ' of ' num2str(length(fileName)) '.']);
            [~, imgName, ~] = fileparts(fileName{k});
            ff = [pathName fileName{k}];
            info = imfinfo(ff);
            numSections = numel(info);
            %Get the boundary data
            if bndryMode == 2
                coords = csvread(fullfile(BoundaryDir,sprintf('boundary for %s.csv',fileName{k})));
            elseif bndryMode == 3
                bff = fullfile(BoundaryDir,sprintf('mask for %s.tif',fileName{k}));
                bdryImg = imread(bff);
                [B,L] = bwboundaries(bdryImg,4);
                coords = B;%vertcat(B{:,1});
%                  coords = vertcat(B{2:end,1});
            end
            %loop through all sections if image is a stack
            for i = 1:numSections
                if numSections > 1
                    set(infoLabel,'String',sprintf('Processing %s. \n file = %d/%d \n slice = %d/%d.', fileName{k},k,length(fileName),i,numSections));
                elseif numSections == 1
                    set(infoLabel,'String',sprintf('Processing %s. \n file = %d/%d.', fileName{k},k,length(fileName)));
                 
                end
                if numSections > 1
                    IMG = imread(ff,i,'Info',info);
                    set(stackSlide,'Value',i);
                    slider_chng_img(stackSlide,0);
                else
                    IMG = imread(ff);
                end
                if size(IMG,3) > 1
                    if advancedOPT.plotrgbFLAG == 0
                        IMG = rgb2gray(IMG);
                        disp('color image was loaded but converted to grayscale image')
                        img = imadjust(IMG);  % YL: only show the adjusted image, but use the original image for analysis
                    elseif advancedOPT.plotrgbFLAG == 1
                        img = IMG;
                        disp('display color image');
                    end
                end
                figure(guiFig);  set(guiFig, 'name', sprintf('%s, %d/%d, %d x %d',fileName{k},i,numSections,size(IMG,1),size(IMG,2)));
                set(imgAx,'NextPlot','replace');
                imshow(IMG,'Parent',imgAx); drawnow;
              
                if bndryMode == 1 || bndryMode == 2   % csv boundary
                     bdryImg = [];
                     [fibFeat] = processImage(IMG, imgName, outDir, keep, coords, distThresh, makeAssocFlag, makeMapFlag, makeOverFlag, makeFeatFlag, i, infoLabel, bndryMode, bdryImg, pathName, fibMode, advancedOPT,numSections);
                else %bndryMode = 3  tif boundary
                     
                     [fibFeat] = processImage(IMG, imgName, outDir, keep, coords, distThresh, makeAssocFlag, makeMapFlag, makeOverFlag, makeFeatFlag, i, infoLabel, bndryMode, bdryImg, pathName, fibMode, advancedOPT,numSections);
                end
                
                if numSections > 1
                    set(infoLabel,'String',sprintf('Done with %s. \n file = %d/%d \n slice = %d/%d.', fileName{k},k,length(fileName),i,numSections));
                elseif numSections == 1
                    set(infoLabel,'String',sprintf('Done with %s. \n file = %d/%d.', fileName{k},k,length(fileName)));
                 
                end
            end
        end
         %Add an option to display the previous analysis results in "CA_Out" folder
         CAout_found = checkCAoutput(pathName,fileName);
         existing_ind = find(cellfun(@isempty, CAout_found) == 0); % index of images with existing output
         if isempty(existing_ind)
             disp('No result was found at "CA_Out" folder. Check/reset the parameters to start over.')
         else
             
%              CA_OLfig_h = findobj(0,'Name','CurveAlign Fiber Overlay');
%              CA_MAPfig_h = findobj(0,'Name','CurveAlign Angle Map');
             CA_HISTfig_h = findobj(0,'Name','Histogram of the angles');
%              if ~isempty(CA_OLfig_h)
%                  close(CA_OLfig_h)
%                  disp('The CurveAlign overlay figure is closed')
%              end
%              if ~isempty(CA_MAPfig_h)
%                  close(CA_MAPfig_h)
%                  disp('The CurveAlign Angle heatmap is closed')
%              end
             if ~isempty(CA_HISTfig_h)
                 close(CA_HISTfig_h)
                 disp('The CurveAlign Angle histogram is closed')
             end
             checkCAout_display_fn(pathName,fileName,existing_ind);
             disp('Click the item in the output table to display the output images')
             figure(CA_table_fig)
             disp(sprintf('Analysis is done. CurveAlign results found at "CA_Out" folder for %d out of %d opened image(s)',...
                 length(existing_ind),length(fileName)))
             note_temp1 = 'Click each item in the output table to display the output images.';
             note_temp2 = 'Running "CurveAlign" here will overwrite them.';
             set(infoLabel, 'String',sprintf('Existing CurveAlign results listed:%d out of %d opened image(s). \n%s \n %s',...
                 length(existing_ind),length(fileName),note_temp1,note_temp2))
         end
     end
%--------------------------------------------------------------------------
% keypress function for the main gui window
    function startPoint(guiFig,evnt)
        if strcmp(evnt.Key,'alt')
            altkey = 1;
            set(guiFig,'WindowKeyReleaseFcn',@stopPoint)
            set(guiFig,'WindowButtonDownFcn',@getPoint)
            set(guiFig,'Pointer','custom','PointerShapeCData',PointerShapeData,'PointerShapeHotSpot',[8,8]);
            
        end
    end
%--------------------------------------------------------------------------
% boundary creation function that records the user's mouse clicks while the
% alt key is being held down
    function getPoint(guiFig,evnt2)
        
        figSize = get(guiFig,'Position');
        aspectImg = imgSize(2)/imgSize(1); %horiz/vert
        aspectFig = figSize(3)/figSize(4); %horiz/vert
        if aspectImg < aspectFig
            %vert limiting dimension
            scaleImg = figSize(4)/imgSize(1);
            vertOffset = 0;
            horizOffset = round((figSize(3) - scaleImg*imgSize(2))/2);
        else
            %horiz limiting dimension
            scaleImg = figSize(3)/imgSize(2);
            vertOffset = round((figSize(4) - scaleImg*imgSize(1))/2);
            horizOffset = 0;
        end
        
        if ~get(guiFig,'UserData')
            coords(aa,:) = get(guiFig,'CurrentPoint');
            %convert the selected point from guiFig coords to actual image
            %coordinages
            curRow = round((figSize(4)-(coords(aa,2) + vertOffset))/scaleImg);
            curCol = round((coords(aa,1) - horizOffset)/scaleImg);
            rows(aa) = curRow;
            cols(aa) = curCol;
            disp(sprintf('current cursor position is [%d %d]', curRow, curCol));
            aa = aa + 1;
            
            figure(guiFig);
            hold on;
            ca = get(guiFig,'CurrentAxes');
            plot(ca,cols,rows,'r');
            plot(ca,cols,rows,'*y');
            %plot(ca,50,50,'r');
            %plot(ca,50,50,'*y');
            
            setappdata(guiFig,'rows',rows);
            setappdata(guiFig,'cols',cols);
        end
    end

%--------------------------------------------------------------------------
% terminates boundary creation when the alt key is released
    function stopPoint(guiFig,evnt4)
        
        if altkey == 1
        set(guiFig,'UserData',1)
        set(guiFig,'WindowButtonUpFcn',[])
        set(guiFig,'WindowKeyPressFcn',[])
        set(guiFig,'WindowButtonDownFcn',[])  %yl
        setappdata(guiFig,'boundary',1)
        coords(:,2) = getappdata(guiFig,'rows');
        coords(:,1) = getappdata(guiFig,'cols');
%         set([enterKeep enterDistThresh makeAngle makeRecon],'Enable','on')
        set([enterKeep enterDistThresh],'Enable','on')
        set(guiFig,'Pointer','default');
        set(makeAssoc,'Enable','on');
        set(enterDistThresh,'Enable','on');
        fileName2 = sprintf('boundary for %s.csv',fileName{index_selected});
        fName = fullfile(BoundaryDir,fileName2);
        csvwrite(fName,coords);
        disp(sprintf('csv boundary for %s was created, set parameters and click Run button to proceed',fileName{index_selected}))
        fprintf('csv boundary coordinates is saved at %s \n',fName) 
        if bndryMode == 2
            set(infoLabel,'string',sprintf('csv boundary for %s was created. Set parameters and click Run button to proceed.',fileName{index_selected}))
            set(imgRun,'Enable','on')
        else
            set(imgRun,'Enable','off')
            set(infoLabel,'string',sprintf('csv boundary for %s was created. To use this boundary: Reset and set boundary mode to CSV boundary.',fileName{index_selected}))
        end
        rows = [];
        cols = [];
        coords = [-1000 -1000];
        aa= 1;
        set(guiFig,'CurrentPoint',coords);
        setappdata(guiFig,'rows',rows);
        setappdata(guiFig,'rows',cols);
        altkey = 0;
        end
        
    end
%--------------------------------------------------------------------------
%Display analyzed oriention and alignment calculated by CurveAlign at
%"CA_Out" folder
    function checkCAout_display_fn(pathName,fileName,existing_ind)
        ii = 0;
        items_number_current = 0;
        CA_data_current = [];
        selectedROWs = [];
        savepath = fullfile(pathName,'CA_Out');
        
        % check the availability of output table
        CA_tablefig_find = findobj(0,'Name', 'CurveAlign output table');
        if isempty(CA_tablefig_find)
            CA_table_fig = figure('Units','normalized','Position',figPOS,'Visible','off',...
                'NumberTitle','off','name','CurveAlign output table');
            CA_output_table = uitable('Parent',CA_table_fig,'Units','normalized','Position',[0.05 0.05 0.9 0.9],...
                'Data', CA_data_current,...
                'ColumnName', columnname,...
                'ColumnFormat', columnformat,...
                'ColumnWidth',columnwidth,...
                'ColumnEditable', [false false false false false false false false false false false false false false],...
                'RowName',[],...
                'CellSelectionCallback',{@CAot_CellSelectionCallback});
        end
        % end of output table check
        figure(CA_table_fig)
        for jj = 1: length(existing_ind)
            [~,imagenameNE] = fileparts(fileName{existing_ind(jj)});
            numSEC = numel(imfinfo(fullfile(pathName,fileName{existing_ind(jj)}))); % 1:single stack; > 1: stack
            if numSEC == 1 % single image
                OLname = fullfile(savepath,[imagenameNE,'_overlay.tiff']);
                filenameALI = fullfile(savepath,sprintf('%s_stats.csv',imagenameNE));      % ctFIRE output:csv angle histogram values
                if ~exist(filenameALI,'file')
                    disp(sprintf('%s not exist, overall alignment not exist', filenameALI))
                    return
                else
%                     disp(sprintf('%s exists, overall alignment will be output', filenameALI))
                    statsOUT = importdata(filenameALI,'\t');
                    IMGangle = nan; IMGali = nan;
                    try   % backwards compatibility
                        IMGangle = statsOUT{1};
                        IMGali = statsOUT{5};
                    catch
                        IMGangle = statsOUT.data(1);
                        IMGali = statsOUT.data(5);
                    end
                end
                % Extract information from the feature .mat file
                matData_ca = fullfile(savepath,[imagenameNE '_fibFeatures.mat']);
                csvANG_ca =  fullfile(savepath,[imagenameNE '_values.csv']);
                if exist(matData_ca, 'file')
                    matTemp = load(matData_ca,'bndryMeas', 'fibProcMeth','fibFeat');
                    if matTemp.fibProcMeth == 0 % "curvelets"
                        modeID = 'Curvelets';
                    else %"CTF fibers" 1,2,3
                        modeID = 'CTF Fibers';
                    end
                    if matTemp.bndryMeas == 0
                        bndryID = 'NO';
                    elseif matTemp.bndryMeas == 1|| matTemp.bndryMeas == 2 || matTemp.bndryMeas == 3 % debug
                        bndryID = 'YES';
                    end
                    if strcmp(bndryID,'YES')
                        fibNUM = size(importdata(csvANG_ca),1);
                    else
                        fibNUM = size(matTemp.fibFeat,1);
                    end
                    clear matTemp;
                end
                xc = nan; yc = nan; zc = 1;
                items_number_current = items_number_current+1;
                CA_data_add = {items_number_current,sprintf('%s',imagenameNE),'',...
                    sprintf('%.1f',IMGangle),sprintf('%.2f',IMGali),sprintf('%d',fibNUM),...
                    modeID,bndryID,'','','',xc,yc,zc,};
                CA_data_current = [CA_data_current;CA_data_add];
                set(CA_output_table,'Data',CA_data_current)
            elseif numSEC > 1   % stack
                for kk = 1:numSEC
                    OLname = fullfile(savepath,[imagenameNE,'_s',num2str(kk),'_overlay.tiff']);
                    filenameALI = fullfile(savepath,sprintf('%s_s%d_stats.csv',imagenameNE,kk));      % ctFIRE output:csv angle histogram values
                    if ~exist(filenameALI,'file')
                        disp(sprintf('%s not exist, overall alignment not exist', filenameALI))
                        return
                    else
                        disp(sprintf('%s exists, overall alignment will be output', filenameALI))
                        statsOUT = importdata(filenameALI,'\t');
                        try   % backwards compatibility
                            IMGangle = statsOUT{1};
                            IMGali = statsOUT{5};
                        catch
                            IMGangle = statsOUT.data(1);
                            IMGali = statsOUT.data(5);
                        end
                        % Extract information from the feature .mat file
                        matData_ca = fullfile(savepath,sprintf('%s_s%d_fibFeatures.mat',imagenameNE,kk));
                        csvANG_ca = fullfile(savepath,sprintf('%s_s%d_values.csv',imagenameNE,kk));
                        if exist(matData_ca, 'file')
                            matTemp = load(matData_ca,'bndryMeas', 'fibProcMeth','fibFeat');
                            if matTemp.fibProcMeth == 0 % "curvelets"
                                modeID = 'Curvelets';
                            else %"CTF fibers" 1,2,3
                                modeID = 'CTF Fibers';
                            end
                            if matTemp.bndryMeas == 0
                                bndryID = 'NO';
                            elseif matTemp.bndryMeas == 2 || matTemp.bndryMeas == 3
                                bndryID = 'YES';
                            end
                            if strcmp(bndryID,'YES')
                                fibNUM = size(importdata(csvANG_ca),1);
                            else
                                fibNUM = size(matTemp.fibFeat,1);
                            end
                            clear matTemp;
                        end
                        xc = nan; yc = nan; zc = kk;
                        items_number_current = items_number_current+1;
                        CA_data_add = {items_number_current,sprintf('%s',imagenameNE),'',...
                            sprintf('%.1f',IMGangle),sprintf('%.2f',IMGali),sprintf('%d',fibNUM),...
                            modeID,bndryID,'','','',xc,yc,zc,};
                        CA_data_current = [CA_data_current;CA_data_add];
                        set(CA_output_table,'Data',CA_data_current)
                    end
                end % slices loop
            end  % single image or stack
        end % image index
    end % checkCAout... function
%--------------------------------------------------------------------------
% returns the user to the measurement selection window
    function resetImg(resetClear,eventdata)
        CurveAlign
    end

    function[]=roi_mang_keypress_fn(object,eventdata,handles)
        % use key press function to quit boundary creation loop
        if BDCchoice == 1  % 1: tiff boundary
            if(eventdata.Key=='m')
                double_click = 1;
                disp('Click(freehand-mode) or Double click(polygon-mode) any point on the image to complete the boundary mask creation')
            end
        end
    end

    function[]=feature_ranking_automation_fn()
        [filesPositiveTemp pathNameTemp] = uigetfile({'*_fibFeatures.mat';'*.*'},'Select Positive annotated image(s)',fullfile(pathNameGlobal),'MultiSelect','on');
        if ~isempty(pathNameTemp)
            pathNameGlobal= pathNameTemp; 
        else
            disp('NO file is selected for feature ranking');
            return
        end
        annotationData=[];
        if(~iscell(filesPositiveTemp))
            filesPositive{1,1}=filesPositiveTemp;
        else
            filesPositive=filesPositiveTemp; 
        end
        for i=1:size(filesPositive,2)
            display(filesPositive{1,i});
            annotationData{i,1}=1;
            annotationData{i,2}=filesPositive{1,i};
            annotationData{i,2}=annotationData{i,2}(1:strfind(annotationData{i,2},'_fibFeatures.mat')-1);
            if(exist(fullfile(pathNameTemp,filesPositive{1,i}),'file')~=2)
                error('file not present');
            end
        end
        [filesNegativeTemp pathNameTemp] = uigetfile({'*_fibFeatures.mat';'*.*'},'Select Negative annotated image(s)',fullfile(pathNameGlobal),'MultiSelect','on');
        if(~iscell(filesNegativeTemp))
            filesNegative{1,1}=filesNegativeTemp;
        else
            filesNegative=filesNegativeTemp; 
        end
        for i=1+size(filesPositive,2):size(filesNegative,2)+size(filesPositive,2)
            display(filesNegative{1,i-size(filesPositive,2)});
            annotationData{i,1}=0;
            annotationData{i,2}=filesNegative{1,i-size(filesPositive,2)};
            annotationData{i,2}=annotationData{i,2}(1:strfind(annotationData{i,2},'_fibFeatures.mat')-1);
            if(exist(fullfile(pathNameTemp,filesNegative{1,i-size(filesPositive,2)}),'file')~=2)
                error('file not present');
            end
        end
        kipper=1;
        size(unique(annotationData(:,2)),1)
        size(annotationData(:,2),1)
        if(size(unique(annotationData(:,2)),1)~=size(annotationData(:,2),1))
           error('overlapping negative and positive images'); 
        end
        display(['annotation data is created and saved at:' fullfile(pathNameTemp,'annotation.xlsx')]);
        try
            xlswrite([fullfile(pathNameTemp)  'annotation.xlsx'],annotationData);
        catch
            xlwrite([fullfile(pathNameTemp)  'annotation.xlsx'],annotationData);
        end
    end

    function[x_min,y_min,x_max,y_max]=enclosing_rect_fn(coordinates)
        x_min=round(min(coordinates(:,1)));
        x_max=round(max(coordinates(:,1)));
        y_min=round(min(coordinates(:,2)));
        y_max=round(max(coordinates(:,2)));
    end


end
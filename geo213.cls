%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{geo213}[2016/08/10 GRAD.GEOF./UFBA class]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                   
%    ___  _ __   ___ ___   ___  ___ 
%   / _ \| '_ \ / __/ _ \ / _ \/ __|
%  | (_) | |_) | (_| (_) |  __/\__ \
%   \___/| .__/ \___\___/ \___||___/
%        |_|                        
% Opcao para produzir a capa com marcas da ANP/FINEP
\newcounter{anpcounter}\setcounter{anpcounter}{0}
\DeclareOption{anp}{\setcounter{anpcounter}{1}}
%
% Opcao para produzir a capa com marcas da PETROBRAS
\newcounter{petrocounter}\setcounter{petrocounter}{0}
\DeclareOption{petro}{\setcounter{petrocounter}{1}}
%
% Opcao para produzir a capa e folha de rosto de um projeto
\newcounter{projetocounter}\setcounter{projetocounter}{0}
\DeclareOption{projeto}{\setcounter{projetocounter}{1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
%
\ProcessOptions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
\LoadClass[a4paper]{report}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Carrega os pacotes
\RequirePackage{pppgbib}
\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{fancyhdr}
\RequirePackage{caption}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Carrega as definições do projeto ou tfg
\input{geo213v1.def}
\ifnum\c@projetocounter=1
	\input{geo213prj1.def}
\else
	\input{geo213tfg1.def}
\fi
%




%% Evaluate the Lab 6a baseline and the damping-randomized SAC agent.
% Edit these three paths when evaluating a different training run.
repoDir = fileparts(mfilename('fullpath'));
runDir = fullfile(repoDir, 'runs', '20261126_202331_dampingArm_pm10pct');
baselineFile = fullfile(repoDir, '..', '..', 'lab6a', ...
    'EECS128_RL_Lab_Rotary_Inverted_Pendulum_Swingup-main', 'Outout', 'experiment_1.mat');
trainedFile = fullfile(runDir, 'checkpoints', 'Agent500.mat');
outputDir = fullfile(runDir, 'evaluation');
numStarts = 5;

assert(isfile(baselineFile) && isfile(trainedFile), 'An agent file is missing.');
if ~isfolder(outputDir), mkdir(outputDir); end

% Same model parameters and initial-state distribution as trainingScript.m.
q1 = 1; q2 = 1; q3 = 0.01; q4 = 0.01; q5 = 2; q6 = 20;
theta_limit = pi/2; volt_limit = 6; Ts = 0.01;
motorResistance = 0.22; dampingArm = 0.25;
dampingPendulum = 0.0007; armCoulomb = 1e-6;
armBreakaway = armCoulomb; velTau = 0.05;
theta0 = 0; phi0 = pi; dtheta0 = 0; dphi0 = 0;

mdl = 'rotpen_rl';
load_system('QuanserQube');
set_param('QuanserQube', 'Lock', 'off');
set_param('QuanserQube/Qube servo/Pendulum Physical Model/Revolute Joint', ...
    'DampingCoefficient_conf', 'runtime');
set_param('QuanserQube', 'Lock', 'on');
load_system(mdl);
set_param(mdl, 'SimulationMode', 'accelerator', 'FastRestart', 'on', ...
    'SimMechanicsOpenEditorOnUpdate', 'off');
env = rlSimulinkEnv(mdl, [mdl '/RL Agent'], rlNumericSpec([7 1]), ...
    rlNumericSpec([1 1], UpperLimit=1, LowerLimit=-1));
simOpts = rlSimulationOptions(MaxSteps=round(5/Ts));

baselineData = load(baselineFile, 'agent');
trainedData = load(trainedFile, 'saved_agent');
agents = {baselineData.agent, trainedData.saved_agent};
names = {'baseline', 'dampingArm_pm10pct'};
for k = 1:2, agents{k}.UseExplorationPolicy = false; end

rng(0, 'twister');
thetaStarts = (2*rand(1,numStarts)-1)*pi/4;
phiStarts = pi + (2*rand(1,numStarts)-1)*pi/4;
dampingValues = [0.225 0.25 0.275];
tol = 15*pi/180;
rows = cell(numel(dampingValues)*numStarts*2, 9);
row = 0;
for d = dampingValues
    for trial = 1:numStarts
        for k = 1:2
            env.ResetFcn = @(in) resetCase(in, thetaStarts(trial), phiStarts(trial), d);
            exp = sim(agents{k}, env, simOpts);
            obs = exp.Observation.obs1;
            t = obs.Time(:);
            phi = atan2(squeeze(obs.Data(4,1,:)), squeeze(obs.Data(5,1,:)));
            outside = find(abs(phi) > tol, 1, 'last');
            if isempty(outside), stayTime = t(1);
            elseif outside == numel(t), stayTime = NaN;
            else, stayTime = t(outside+1);
            end
            success = t(end) >= 5-Ts && ~isnan(stayTime) && stayTime <= 2;
            row = row+1;
            rows(row,:) = {names{k},d,trial,thetaStarts(trial),phiStarts(trial), ...
                success,stayTime,t(end),rad2deg(phi(end))};
            trace = table(t, rad2deg(phi), repmat(d,numel(t),1), ...
                'VariableNames', {'time_s','pendulum_angle_deg','dampingArm'});
            writetable(trace, fullfile(outputDir, sprintf('angle_%s_d%.3f_trial%d.csv', ...
                names{k},d,trial)));
            fprintf('%s d=%.3f trial=%d success=%d stay=%.2f s\n', ...
                names{k},d,trial,success,stayTime);
        end
    end
end

results = cell2table(rows, 'VariableNames', ...
    {'agent','dampingArm','trial','theta0','phi0','success', ...
     'sustainedEntry_s','simulationEnd_s','finalAngle_deg'});
writetable(results, fullfile(outputDir, 'evaluation_results.csv'));
save(fullfile(outputDir, 'evaluation_results.mat'), 'results', 'tol', ...
    'thetaStarts', 'phiStarts', 'dampingValues');

rates = zeros(numel(dampingValues),2);
for i = 1:numel(dampingValues)
    for k = 1:2
        rates(i,k) = mean(results.success(results.dampingArm == dampingValues(i) ...
            & strcmp(results.agent,names{k})))*100;
    end
end
fig = figure('Visible','off');
bar(dampingValues, rates, 'grouped'); ylim([0 105]); grid on;
xticks(dampingValues); xticklabels(compose('%.3f', dampingValues));
xlabel('Arm damping (N m s/rad)'); ylabel('Success rate (%)');
legend(names, 'Location','best'); title('Swing-up and hold through 5 s');
savefig(fig, fullfile(outputDir, 'success_rates.fig'));
exportgraphics(fig, fullfile(outputDir, 'success_rates.png'));
close(fig);

set_param(mdl, 'FastRestart', 'off');
close_system(mdl, 0);
close_system('QuanserQube', 0);

function in = resetCase(in, theta, phi, damping)
in = setVariable(in, 'theta0', theta);
in = setVariable(in, 'phi0', phi);
in = setVariable(in, 'dtheta0', 0);
in = setVariable(in, 'dphi0', 0);
in = setVariable(in, 'dampingArm', damping);
end

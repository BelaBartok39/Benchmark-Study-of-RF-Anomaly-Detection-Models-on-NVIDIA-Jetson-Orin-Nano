%% Detailed APM Analysis - Additional Figures for Paper Extension
% This script generates supplementary figures for in-depth analysis

clear; clc; close all;

%% Data Setup - Single measurements per configuration
models = {'AAE', 'AE', 'CNN-AE', 'LSTM-AE', 'ResNet-AE'};
workloads = {'Bursty', 'Periodic', 'Continuous', 'Variable'};

% Total Energy Consumption (J) - Static MAXN
energy_static = [
     9513.23,  9101.70,  9761.26,  9802.85;   % AAE
     9568.38,  9149.15, 10030.03, 10123.07;   % AE
     9584.05,  9104.84, 10215.72, 10348.29;   % CNN-AE
     9613.47,  9110.09, 10295.42, 10682.78;   % LSTM-AE
     9577.33,  9096.89, 10297.42, 10676.89;   % ResNet-AE
];

% Total Energy Consumption (J) - Adaptive
energy_adaptive = [
     9440.66,  8995.33,  9687.84,  9729.32;   % AAE
     9501.15,  9032.72,  9946.60, 10078.08;   % AE
     9503.83,  9004.06, 10179.73, 10280.39;   % CNN-AE
     9546.34,  9009.96, 10072.30, 10448.65;   % LSTM-AE
     9596.46,  9044.82, 10312.29, 10715.67;   % ResNet-AE
];

% Mode switches across all workloads
mode_switches_all = [
    1, 1, 3, 4;     % AAE
    4, 1, 1, 6;     % AE
    21, 9, 34, 42;  % CNN-AE
    3, 48, 50, 44;  % LSTM-AE
    2, 51, 3, 5     % ResNet-AE
];

% Power mode distribution for Bursty workload [15W, 25W, MAXN]
mode_dist_bursty = [
    100.0, 0.0, 0.0;    % AAE
    96.7, 1.7, 1.7;     % AE
    81.9, 16.4, 1.7;    % CNN-AE
    0.0, 65.3, 34.7;    % LSTM-AE
    0.0, 0.0, 100.0     % ResNet-AE
];

% Power mode distribution for Continuous workload
mode_dist_continuous = [
    98.3, 1.7, 0.0;     % AAE
    100.0, 0.0, 0.0;    % AE
    4.5, 35.2, 60.3;    % CNN-AE
    53.4, 46.6, 0.0;    % LSTM-AE
    0.0, 0.0, 99.9      % ResNet-AE
];

% FPS per Watt improvement (%)
fps_per_watt_improvement = [
    0.8, 1.2, 0.8, 0.8;   % AAE
    0.7, 1.3, 0.8, 0.5;   % AE
    0.9, 1.1, 0.4, 0.7;   % CNN-AE
    0.7, 1.1, 2.2, 2.2;   % LSTM-AE
   -0.2, 0.6,-0.1,-0.4    % ResNet-AE
];

%% Figure 1: Total Energy Consumption - Static vs Adaptive
figure('Position', [100 100 1400 600]);

for i = 1:2
    subplot(1,2,i);
    if i == 1
        data = energy_static;
        title_str = 'Static MAXN Mode';
    else
        data = energy_adaptive;
        title_str = 'Adaptive Power Management';
    end

    bar(data');
    set(gca, 'XTickLabel', workloads);
    legend(models, 'Location', 'northwest', 'FontSize', 10);
    ylabel('Total Energy (J)', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Workload Pattern', 'FontSize', 12, 'FontWeight', 'bold');
    title(title_str, 'FontSize', 13, 'FontWeight', 'bold');
    grid on;
    ylim([8500 11000]);
    set(gca, 'FontSize', 11);
end

sgtitle('Total Energy Consumption Comparison', 'FontSize', 15, 'FontWeight', 'bold');
saveas(gcf, 'total_energy_comparison.png');
saveas(gcf, 'total_energy_comparison.fig');

%% Figure 2: Mode Switching Behavior
figure('Position', [100 100 1200 700]);

% Create heatmap of mode switches
imagesc(mode_switches_all);
colormap(hot);
colorbar;
caxis([0 55]);

set(gca, 'XTick', 1:4, 'XTickLabel', workloads);
set(gca, 'YTick', 1:5, 'YTickLabel', models);
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
title('Power Mode Switching Frequency', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Workload Pattern', 'FontSize', 13);
ylabel('Model Architecture', 'FontSize', 13);

% Add text annotations
for i = 1:5
    for j = 1:4
        if mode_switches_all(i,j) > 30
            text_color = 'white';
        else
            text_color = 'black';
        end
        text(j, i, sprintf('%d', mode_switches_all(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 11, ...
            'FontWeight', 'bold', 'Color', text_color);
    end
end

grid on;
saveas(gcf, 'mode_switching_heatmap.png');
saveas(gcf, 'mode_switching_heatmap.fig');

%% Figure 3: Workload-Specific Power Mode Distribution
figure('Position', [100 100 1400 700]);

% Bursty workload
subplot(1,2,1);
bar(mode_dist_bursty, 'stacked');
set(gca, 'XTickLabel', models);
legend({'15W', '25W', 'MAXN'}, 'Location', 'eastoutside', 'FontSize', 10);
ylabel('Time in Mode (%)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Model Architecture', 'FontSize', 12, 'FontWeight', 'bold');
title('Bursty Workload', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
ylim([0 100]);
set(gca, 'FontSize', 11);

% Continuous workload
subplot(1,2,2);
bar(mode_dist_continuous, 'stacked');
set(gca, 'XTickLabel', models);
legend({'15W', '25W', 'MAXN'}, 'Location', 'eastoutside', 'FontSize', 10);
ylabel('Time in Mode (%)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Model Architecture', 'FontSize', 12, 'FontWeight', 'bold');
title('Continuous Workload', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
ylim([0 100]);
set(gca, 'FontSize', 11);

sgtitle('Power Mode Distribution by Workload Type', 'FontSize', 15, 'FontWeight', 'bold');
saveas(gcf, 'workload_specific_modes.png');
saveas(gcf, 'workload_specific_modes.fig');

%% Figure 4: FPS per Watt Improvement
figure('Position', [100 100 1200 600]);
bar(fps_per_watt_improvement');
set(gca, 'XTickLabel', workloads);
legend(models, 'Location', 'northwest', 'FontSize', 11);
ylabel('FPS/Watt Improvement (%)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Workload Pattern', 'FontSize', 13, 'FontWeight', 'bold');
title('Energy Efficiency Improvement (FPS per Watt)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12);
yline(0, 'k--', 'LineWidth', 1.5);
ylim([-0.5 2.5]);

saveas(gcf, 'fps_per_watt_improvement.png');
saveas(gcf, 'fps_per_watt_improvement.fig');

%% Figure 5: Energy Savings by Workload (Box Plot Style)
figure('Position', [100 100 1000 700]);

% Compute energy savings
energy_savings_calc = ((energy_static - energy_adaptive) ./ energy_static) * 100;

% Create grouped data for box plot
all_savings = energy_savings_calc';
boxplot(all_savings, 'Labels', models);
ylabel('Energy Savings (%)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Model Architecture', 'FontSize', 13, 'FontWeight', 'bold');
title('Distribution of Energy Savings Across Workloads', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12);
yline(0, 'k--', 'LineWidth', 1.5);

saveas(gcf, 'energy_savings_boxplot.png');
saveas(gcf, 'energy_savings_boxplot.fig');

%% Figure 6: Model Efficiency Classification
figure('Position', [100 100 1000 700]);

% Classify models based on APM behavior
avg_savings = mean(energy_savings_calc, 2);
avg_switches = mean(mode_switches_all, 2);

% Create scatter with size proportional to switches
scatter(avg_savings, [1:5], avg_switches*10, ...
    [0.2 0.6 0.8; 0.8 0.4 0.2; 0.4 0.8 0.2; 0.8 0.2 0.6; 0.9 0.7 0.1], ...
    'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);

% Add labels
for i = 1:5
    text(avg_savings(i)+0.05, i, sprintf('%s (%.0f switches)', models{i}, avg_switches(i)), ...
        'FontSize', 11, 'FontWeight', 'bold');
end

xlabel('Average Energy Savings (%)', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'YTick', 1:5, 'YTickLabel', models);
ylabel('Model Architecture', 'FontSize', 13, 'FontWeight', 'bold');
title('Model Efficiency Profile (Bubble size = Avg. Mode Switches)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
xline(0, 'k--', 'LineWidth', 1.5);
set(gca, 'FontSize', 12);

saveas(gcf, 'efficiency_classification.png');
saveas(gcf, 'efficiency_classification.fig');

%% Generate Comprehensive Summary Table
fprintf('\n=== COMPREHENSIVE SUMMARY TABLE (LATEX) ===\n\n');
fprintf('\\begin{table*}[t]\n');
fprintf('\\centering\n');
fprintf('\\caption{Comprehensive APM Performance Summary Across All Workloads}\n');
fprintf('\\label{tab:apm_comprehensive}\n');
fprintf('\\small\n');
fprintf('\\begin{tabular}{lccccc}\n');
fprintf('\\hline\n');
fprintf('Model & Avg Energy & Avg Power & Avg FPS/W & Avg Mode & Primary \\\\\n');
fprintf(' & Savings (\\%%) & Reduction (W) & Improvement (\\%%) & Switches & Power Mode \\\\\n');
fprintf('\\hline\n');

for i = 1:5
    avg_power_reduction = mean(energy_static(i,:) - energy_adaptive(i,:))/1800;  % Average power saved
    avg_fps_improvement = mean(fps_per_watt_improvement(i,:));

    % Determine primary mode
    if i == 1 || i == 2
        primary_mode = '15W';
    elseif i == 5
        primary_mode = 'MAXN';
    else
        primary_mode = 'Mixed';
    end

    fprintf('%s & %.2f & %.2f & %.2f & %.0f & %s \\\\\n', ...
        models{i}, avg_savings(i), avg_power_reduction, ...
        avg_fps_improvement, avg_switches(i), primary_mode);
end

fprintf('\\hline\n');
fprintf('\\end{tabular}\n');
fprintf('\\end{table*}\n\n');

fprintf('=== Detailed Analysis Complete ===\n');
fprintf('Generated additional files:\n');
fprintf('  - total_energy_comparison.png/.fig\n');
fprintf('  - mode_switching_heatmap.png/.fig\n');
fprintf('  - workload_specific_modes.png/.fig\n');
fprintf('  - fps_per_watt_improvement.png/.fig\n');
fprintf('  - energy_savings_boxplot.png/.fig\n');
fprintf('  - efficiency_classification.png/.fig\n');

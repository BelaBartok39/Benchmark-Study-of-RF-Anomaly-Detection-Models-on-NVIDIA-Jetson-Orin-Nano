%% Adaptive Power Management Study - Figure Generation for Paper Extension
% This script generates figures and tables summarizing APM results
% for the paper extension section on "Increasing Efficiency"

clear; clc; close all;

%% Data: Energy Savings (%) by Model and Workload
% Positive values indicate energy savings with APM
% Data extracted from JSON files - single measurement per configuration
models = {'AAE', 'AE', 'CNN-AE', 'LSTM-AE', 'ResNet-AE'};
workloads = {'Bursty', 'Periodic', 'Continuous', 'Variable'};

% Energy savings data (%)
energy_savings = [
      0.76,   1.17,   0.75,   0.75;   % AAE
      0.70,   1.27,   0.83,   0.44;   % AE
      0.84,   1.11,   0.35,   0.66;   % CNN-AE
      0.70,   1.10,   2.17,   2.19;   % LSTM-AE
     -0.20,   0.57,  -0.14,  -0.36;   % ResNet-AE
];

% Average power consumption - Static MAXN (W)
power_static = [
      5.28,   5.26,   5.42,   5.44;   % AAE
      5.31,   5.29,   5.57,   5.62;   % AE
      5.32,   5.26,   5.67,   5.75;   % CNN-AE
      5.34,   5.27,   5.72,   5.93;   % LSTM-AE
      5.32,   5.26,   5.72,   5.93;   % ResNet-AE
];

% Average power consumption - Adaptive (W)
power_adaptive = [
      5.24,   5.20,   5.38,   5.40;   % AAE
      5.28,   5.22,   5.52,   5.60;   % AE
      5.28,   5.20,   5.65,   5.71;   % CNN-AE
      5.30,   5.21,   5.59,   5.80;   % LSTM-AE
      5.33,   5.23,   5.73,   5.95;   % ResNet-AE
];

% Power mode distribution (%) - Time spent in each mode for Variable workload
% [15W, 25W, MAXN]
mode_distribution_variable = [
    94.4, 1.7, 3.9;    % AAE
    95.0, 3.3, 1.7;    % AE
    10.7, 48.0, 41.3;  % CNN-AE
    28.1, 71.9, 0.0;   % LSTM-AE
    0.8, 0.0, 99.2     % ResNet-AE
];

% Mode switches for Variable workload
mode_switches_variable = [4, 6, 42, 44, 5];

% SLA violation rates - Static vs Adaptive for Variable workload (%)
sla_violations = [
    0.00, 0.00;   % AAE
    0.00, 0.00;   % AE
    0.00, 0.01;   % CNN-AE
    10.88, 0.00;  % LSTM-AE
    11.70, 10.55  % ResNet-AE
];

%% Figure 1: Energy Savings Heatmap
figure('Position', [100 100 1000 600]);
imagesc(energy_savings);
colormap(jet);
colorbar;
caxis([-0.5 2.5]);

% Labels
set(gca, 'XTick', 1:4, 'XTickLabel', workloads);
set(gca, 'YTick', 1:5, 'YTickLabel', models);
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
title('Energy Savings with Adaptive Power Management (%)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Workload Pattern', 'FontSize', 13);
ylabel('Model Architecture', 'FontSize', 13);

% Add text annotations
for i = 1:5
    for j = 1:4
        if energy_savings(i,j) >= 0
            text(j, i, sprintf('%.2f%%', energy_savings(i,j)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, ...
                'FontWeight', 'bold', 'Color', 'white');
        else
            text(j, i, sprintf('%.2f%%', energy_savings(i,j)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, ...
                'FontWeight', 'bold', 'Color', 'black');
        end
    end
end

grid on;
saveas(gcf, 'energy_savings_heatmap.png');
saveas(gcf, 'energy_savings_heatmap.fig');

%% Figure 2: Energy Savings by Model (Grouped Bar Chart)
figure('Position', [100 100 1200 600]);
bar(energy_savings');
set(gca, 'XTickLabel', workloads);
legend(models, 'Location', 'northwest', 'FontSize', 11);
ylabel('Energy Savings (%)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Workload Pattern', 'FontSize', 13, 'FontWeight', 'bold');
title('Energy Savings Across Models and Workloads', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12);
ylim([-0.5 2.5]);
yline(0, 'k--', 'LineWidth', 1.5);

saveas(gcf, 'energy_savings_bars.png');
saveas(gcf, 'energy_savings_bars.fig');

%% Figure 3: Power Mode Distribution (Stacked Bar Chart)
figure('Position', [100 100 1000 600]);
bar(mode_distribution_variable, 'stacked');
set(gca, 'XTickLabel', models);
legend({'15W (Low Power)', '25W (Medium Power)', 'MAXN (High Power)'}, ...
    'Location', 'eastoutside', 'FontSize', 11);
ylabel('Time in Power Mode (%)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Model Architecture', 'FontSize', 13, 'FontWeight', 'bold');
title('Power Mode Utilization - Variable Workload', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12);
ylim([0 100]);

% Add annotations for mode switches
for i = 1:5
    text(i, 105, sprintf('%d switches', mode_switches_variable(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

saveas(gcf, 'power_mode_distribution.png');
saveas(gcf, 'power_mode_distribution.fig');

%% Figure 4: Average Power Consumption Comparison
figure('Position', [100 100 1200 700]);

% Calculate average across workloads for each model
avg_power_static_all = mean(power_static, 2);
avg_power_adaptive_all = mean(power_adaptive, 2);

% Create grouped bar chart
X = categorical(models);
X = reordercats(X, models);
bar_data = [avg_power_static_all, avg_power_adaptive_all];
b = bar(X, bar_data);

% Customize colors
b(1).FaceColor = [0.8 0.2 0.2];  % Red for static
b(2).FaceColor = [0.2 0.8 0.2];  % Green for adaptive

legend({'Static MAXN', 'Adaptive PM'}, 'Location', 'northwest', 'FontSize', 12);
ylabel('Average Power (W)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Model Architecture', 'FontSize', 13, 'FontWeight', 'bold');
title('Average Power Consumption: Static vs Adaptive', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12);

% Add value labels on bars
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(round(b(1).YData, 2));
text(xtips1, ytips1, labels1, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'FontSize', 10);

xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(round(b(2).YData, 2));
text(xtips2, ytips2, labels2, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'FontSize', 10);

saveas(gcf, 'power_comparison.png');
saveas(gcf, 'power_comparison.fig');

%% Figure 5: SLA Violations Comparison
figure('Position', [100 100 1000 600]);
X = categorical(models);
X = reordercats(X, models);
b = bar(X, sla_violations);

b(1).FaceColor = [0.8 0.2 0.2];  % Red for static
b(2).FaceColor = [0.2 0.8 0.2];  % Green for adaptive

legend({'Static MAXN', 'Adaptive PM'}, 'Location', 'northwest', 'FontSize', 12);
ylabel('SLA Violation Rate (%)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Model Architecture', 'FontSize', 13, 'FontWeight', 'bold');
title('SLA Violation Rates: Variable Workload', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12);

saveas(gcf, 'sla_violations.png');
saveas(gcf, 'sla_violations.fig');

%% Figure 6: Energy Savings vs Model Complexity (Scatter Plot)
figure('Position', [100 100 900 700]);

% Use average energy savings across all workloads
avg_energy_savings = mean(energy_savings, 2);

% Create scatter plot with different markers for each model
model_colors = [
    0.2 0.6 0.8;   % AAE - Blue
    0.8 0.4 0.2;   % AE - Orange
    0.4 0.8 0.2;   % CNN-AE - Green
    0.8 0.2 0.6;   % LSTM-AE - Purple
    0.9 0.7 0.1    % ResNet-AE - Yellow
];

hold on;
for i = 1:5
    scatter(avg_power_static_all(i), avg_energy_savings(i), 200, ...
        model_colors(i,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    text(avg_power_static_all(i)+0.03, avg_energy_savings(i), models{i}, ...
        'FontSize', 11, 'FontWeight', 'bold');
end
hold off;

xlabel('Average Power Consumption (W)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Average Energy Savings (%)', 'FontSize', 13, 'FontWeight', 'bold');
title('Energy Efficiency Gains vs Power Consumption', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12);
yline(0, 'k--', 'LineWidth', 1.5);

saveas(gcf, 'energy_vs_power.png');
saveas(gcf, 'energy_vs_power.fig');

%% Generate Summary Table (for LaTeX)
fprintf('\n=== LATEX TABLE: Energy Savings Summary ===\n\n');
fprintf('\\begin{table}[h]\n');
fprintf('\\centering\n');
fprintf('\\caption{Energy Savings with Adaptive Power Management}\n');
fprintf('\\label{tab:apm_energy_savings}\n');
fprintf('\\begin{tabular}{lcccc}\n');
fprintf('\\hline\n');
fprintf('Model & Bursty & Periodic & Continuous & Variable \\\\\n');
fprintf('\\hline\n');
for i = 1:5
    fprintf('%s', models{i});
    for j = 1:4
        if energy_savings(i,j) >= 0
            fprintf(' & %.1f\\%%', energy_savings(i,j));
        else
            fprintf(' & %.1f\\%%', energy_savings(i,j));
        end
    end
    fprintf(' \\\\\n');
end
fprintf('\\hline\n');
fprintf('\\end{tabular}\n');
fprintf('\\end{table}\n\n');

fprintf('=== LATEX TABLE: Power Mode Distribution (Variable Workload) ===\n\n');
fprintf('\\begin{table}[h]\n');
fprintf('\\centering\n');
fprintf('\\caption{Power Mode Utilization - Variable Workload}\n');
fprintf('\\label{tab:power_mode_dist}\n');
fprintf('\\begin{tabular}{lcccr}\n');
fprintf('\\hline\n');
fprintf('Model & 15W (\\%%) & 25W (\\%%) & MAXN (\\%%) & Switches \\\\\n');
fprintf('\\hline\n');
for i = 1:5
    fprintf('%s & %.1f & %.1f & %.1f & %d \\\\\n', ...
        models{i}, mode_distribution_variable(i,1), ...
        mode_distribution_variable(i,2), mode_distribution_variable(i,3), ...
        mode_switches_variable(i));
end
fprintf('\\hline\n');
fprintf('\\end{tabular}\n');
fprintf('\\end{table}\n\n');

%% Display Summary Statistics
fprintf('=== SUMMARY STATISTICS ===\n\n');
fprintf('Best Energy Savings (Average across workloads):\n');
[max_savings, idx] = max(avg_energy_savings);
fprintf('  %s: %.2f%%\n\n', models{idx}, max_savings);

fprintf('Models with Positive Energy Savings in All Workloads:\n');
for i = 1:5
    if all(energy_savings(i,:) > 0)
        fprintf('  %s\n', models{i});
    end
end

fprintf('\nModels Primarily Using Low Power Mode (>90%% in 15W):\n');
for i = 1:5
    if mode_distribution_variable(i,1) > 90
        fprintf('  %s: %.1f%% in 15W mode\n', models{i}, mode_distribution_variable(i,1));
    end
end

fprintf('\n=== Figure generation complete ===\n');
fprintf('Generated files:\n');
fprintf('  - energy_savings_heatmap.png/.fig\n');
fprintf('  - energy_savings_bars.png/.fig\n');
fprintf('  - power_mode_distribution.png/.fig\n');
fprintf('  - power_comparison.png/.fig\n');
fprintf('  - sla_violations.png/.fig\n');
fprintf('  - energy_vs_power.png/.fig\n');

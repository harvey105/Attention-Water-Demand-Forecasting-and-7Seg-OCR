"""Generate high-resolution PNG and SVG figures for Task 2 and Task 3 using Plotly."""
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIG_DIR = './figures'
TASK2_EVAL = './data/processed/task2_eval.npz'
TASK3_EVAL = './data/processed/task3_eval.npz'
TASK3_IMG_DIR = './data/raw/7seg_images'


def nse_score(y_true, y_pred):
    return 1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def save_fig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.write_image(os.path.join(FIG_DIR, f'{name}.png'), scale=3)
    fig.write_image(os.path.join(FIG_DIR, f'{name}.svg'))
    print(f'Saved {name}.png and {name}.svg')


def plot_task2():
    d = np.load(TASK2_EVAL)
    y_true, y_pred = d['y_true'], d['y_pred']
    loss, val_loss = d['loss'], d['val_loss']
    mae, val_mae = d['mae'], d['val_mae']
    epochs = np.arange(1, len(loss) + 1)
    best_ep = int(np.argmin(val_loss)) + 1

    # 1. Task 2 Training History
    fig1 = make_subplots(rows=1, cols=2, subplot_titles=(
        'Mean Squared Error (Scaled)', 'Mean Absolute Error (Scaled)'
    ))
    fig1.add_trace(go.Scatter(x=epochs, y=loss, name='Train Loss', line=dict(color='#1f77b4', width=2.5)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=epochs, y=val_loss, name='Validation Loss', line=dict(color='#ff7f0e', width=2.5)), row=1, col=1)
    fig1.add_vline(x=best_ep, line_dash='dot', line_color='#2ca02c', annotation_text=f'Best Epoch ({best_ep})', row=1, col=1)

    fig1.add_trace(go.Scatter(x=epochs, y=mae, name='Train MAE', line=dict(color='#1f77b4', width=2.5), showlegend=False), row=1, col=2)
    fig1.add_trace(go.Scatter(x=epochs, y=val_mae, name='Validation MAE', line=dict(color='#ff7f0e', width=2.5), showlegend=False), row=1, col=2)
    fig1.add_vline(x=best_ep, line_dash='dot', line_color='#2ca02c', row=1, col=2)

    fig1.update_layout(
        template='plotly_white', width=1100, height=460,
        title=dict(text='<b>Task 2: Encoder–Decoder LSTM + Attention Training Convergence</b><br><sup>Hourly Water Demand Forecasting (Best Validation Loss = %.4f at Epoch %d)</sup>' % (val_loss[best_ep - 1], best_ep)),
        legend=dict(orientation='h', yanchor='bottom', y=1.06, xanchor='right', x=1.0),
    )
    fig1.update_xaxes(title_text='Training Epoch')
    fig1.update_yaxes(title_text='MSE (z-score)', row=1, col=1)
    fig1.update_yaxes(title_text='MAE (z-score)', row=1, col=2)
    save_fig(fig1, 'task2_training_history')

    # 2. Task 2 Forecast Comparison (168h = 1 week test slice)
    # Align each horizon to its actual target timestamp (window i, step k -> timestamp i + k)
    hours = np.arange(168)
    r2_all = r2_score(y_true.reshape(-1), y_pred.reshape(-1))
    nse_all = nse_score(y_true.reshape(-1), y_pred.reshape(-1))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hours, y=y_true[:168, 0], name='Actual Water Demand', line=dict(color='#1b263b', width=2.5)))
    fig2.add_trace(go.Scatter(x=hours, y=y_pred[:168, 0], name='Predicted Horizon t+1', line=dict(color='#0077b6', width=2, dash='dash')))
    fig2.add_trace(go.Scatter(x=hours[4:], y=y_pred[:164, 4], name='Predicted Horizon t+5 (Aligned)', line=dict(color='#e63946', width=1.8, dash='dot')))
    fig2.update_layout(
        template='plotly_white', width=1100, height=480,
        title=dict(text=f'<b>Task 2: Multi-Step Water Demand Forecast vs. Ground Truth (1-Week Test Window)</b><br><sup>Overall Test Performance across t+1..t+5: R² = {r2_all:.4f}, NSE = {nse_all:.4f}</sup>'),
        xaxis_title='Test Time Index (Hours)',
        yaxis_title='Water Consumption (m³/h)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0),
    )
    save_fig(fig2, 'task2_forecast_comparison')

    # 3. Task 2 Horizon Breakdown + Parity Scatter
    horizons = [f't+{i+1}' for i in range(y_true.shape[1])]
    maes = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    rmses = [root_mean_squared_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    r2s = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]

    fig3 = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.14,
        specs=[[{'secondary_y': True}, {'secondary_y': False}]],
        subplot_titles=('Error & Fit Across Forecast Horizons (t+1 to t+5)', 'Parity Plot: Predicted vs. Actual Demand (Overall)'),
    )
    fig3.add_trace(go.Bar(x=horizons, y=maes, name='MAE (m³/h)', marker_color='#457b9d'), row=1, col=1, secondary_y=False)
    fig3.add_trace(go.Bar(x=horizons, y=rmses, name='RMSE (m³/h)', marker_color='#a8dadc'), row=1, col=1, secondary_y=False)
    fig3.add_trace(go.Scatter(x=horizons, y=r2s, name='R² / NSE', mode='lines+markers+text', text=[f'{v:.3f}' for v in r2s], textposition='top center', textfont=dict(color='#b5171e', size=11), line=dict(color='#e63946', width=2.5), marker=dict(size=8)), row=1, col=1, secondary_y=True)

    yt_flat, yp_flat = y_true.reshape(-1), y_pred.reshape(-1)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(yt_flat), size=min(4000, len(yt_flat)), replace=False)
    fig3.add_trace(go.Scattergl(x=yt_flat[idx], y=yp_flat[idx], mode='markers', name='Test Samples', marker=dict(size=4, opacity=0.35, color='#1d3557'), showlegend=False), row=1, col=2)
    min_v, max_v = float(yt_flat.min()), float(yt_flat.max())
    fig3.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines', name='Ideal Fit (y = x)', line=dict(color='#e63946', dash='dash', width=2)), row=1, col=2)

    fig3.update_layout(
        template='plotly_white', width=1200, height=490, barmode='group',
        title=dict(text=f'<b>Task 2: Multi-Horizon Error Analysis & Prediction Parity (N = {len(y_true):,} Windows)</b><br><sup>Overall MAE = {mean_absolute_error(yt_flat, yp_flat):.2f} m³/h | RMSE = {root_mean_squared_error(yt_flat, yp_flat):.2f} m³/h | R² = NSE = {r2_all:.4f}</sup>'),
        legend=dict(orientation='h', yanchor='bottom', y=1.06, xanchor='right', x=1.0),
    )
    fig3.update_xaxes(title_text='Forecast Horizon', row=1, col=1)
    fig3.update_yaxes(title_text='Error (m³/h)', range=[0, max(rmses) * 1.15], row=1, col=1, secondary_y=False)
    fig3.update_yaxes(title_text='R² / NSE Score', range=[0.85, 1.0], dtick=0.03, tickformat='.2f', row=1, col=1, secondary_y=True)
    fig3.update_xaxes(title_text='Actual Consumption (m³/h)', row=1, col=2)
    fig3.update_yaxes(title_text='Predicted Consumption (m³/h)', row=1, col=2)
    save_fig(fig3, 'task2_horizon_metrics')


def plot_task3():
    d = np.load(TASK3_EVAL, allow_pickle=True)
    y_true = [str(x) for x in d['y_true']]
    y_pred = [str(x) for x in d['y_pred']]
    filenames = [str(x) for x in d['filenames']]
    loss, val_loss = d['loss'], d['val_loss']
    acc, val_acc = d['accuracy'] * 100.0, d['val_accuracy'] * 100.0
    epochs = np.arange(1, len(loss) + 1)
    best_ep = int(np.argmin(val_loss)) + 1

    # 4. Task 3 Training History
    fig4 = make_subplots(rows=1, cols=2, subplot_titles=(
        'Masked Cross-Entropy Loss', 'Token-Level Accuracy (%)'
    ))
    fig4.add_trace(go.Scatter(x=epochs, y=loss, name='Train Loss', line=dict(color='#1f77b4', width=2.5)), row=1, col=1)
    fig4.add_trace(go.Scatter(x=epochs, y=val_loss, name='Validation Loss', line=dict(color='#ff7f0e', width=2.5)), row=1, col=1)
    fig4.add_vline(x=best_ep, line_dash='dot', line_color='#2ca02c', annotation_text=f'Best Epoch ({best_ep})', row=1, col=1)

    fig4.add_trace(go.Scatter(x=epochs, y=acc, name='Train Acc', line=dict(color='#1f77b4', width=2.5), showlegend=False), row=1, col=2)
    fig4.add_trace(go.Scatter(x=epochs, y=val_acc, name='Validation Acc', line=dict(color='#ff7f0e', width=2.5), showlegend=False), row=1, col=2)
    fig4.add_vline(x=best_ep, line_dash='dot', line_color='#2ca02c', row=1, col=2)

    fig4.update_layout(
        template='plotly_white', width=1100, height=460,
        title=dict(text='<b>Task 3: CNN Encoder + LSTM Decoder + Multi-Head Attention Training Convergence</b><br><sup>7SEG_OCR Sequence Recognition (Best Val Loss = %.4f, Val Token Acc = %.2f%% at Epoch %d)</sup>' % (val_loss[best_ep - 1], val_acc[best_ep - 1], best_ep)),
        legend=dict(orientation='h', yanchor='bottom', y=1.06, xanchor='right', x=1.0),
    )
    fig4.update_xaxes(title_text='Training Epoch')
    fig4.update_yaxes(title_text='Sparse Categorical Cross-Entropy', row=1, col=1)
    fig4.update_yaxes(title_text='Token Accuracy (%)', row=1, col=2)
    save_fig(fig4, 'task3_training_history')

    # 5. Task 3 Evaluation Summary (Exact-Match by Length & Error Breakdown)
    df_eval = pd.DataFrame({'true': y_true, 'pred': y_pred})
    df_eval['len'] = df_eval['true'].str.len()
    df_eval['exact'] = (df_eval['true'] == df_eval['pred']).astype(int)
    len_stats = df_eval.groupby('len').agg(acc=('exact', 'mean'), count=('exact', 'count')).reset_index()

    exact_cnt = int(df_eval['exact'].sum())
    dot_err_cnt = sum(
        (t != p) and (t.replace('.', '') == p.replace('.', ''))
        for t, p in zip(y_true, y_pred)
    )
    other_err_cnt = len(y_true) - exact_cnt - dot_err_cnt
    cer = sum(levenshtein_distance(p, t) for p, t in zip(y_pred, y_true)) / sum(len(t) for t in y_true)
    exact_acc = exact_cnt / len(y_true) * 100.0

    fig5 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Exact-Match Accuracy by Sequence Length', 'Test Prediction Outcome Breakdown'),
    )
    fig5.add_trace(go.Bar(
        x=[f'{l} chars<br>(n={c})' for l, c in zip(len_stats['len'], len_stats['count'])],
        y=len_stats['acc'] * 100.0,
        text=[f'{v * 100:.1f}%' for v in len_stats['acc']],
        textposition='outside',
        marker_color='#2a9d8f', name='Exact-Match %',
    ), row=1, col=1)

    fig5.add_trace(go.Bar(
        x=['Exact Match', 'Decimal-Dot Only Error', 'Digit / Sign Error'],
        y=[exact_cnt, dot_err_cnt, other_err_cnt],
        text=[f'{c} ({c/len(y_true)*100:.1f}%)' for c in [exact_cnt, dot_err_cnt, other_err_cnt]],
        textposition='outside',
        marker_color=['#2a9d8f', '#e9c46a', '#e76f51'],
        showlegend=False,
    ), row=1, col=2)

    fig5.update_layout(
        template='plotly_white', width=1120, height=470, showlegend=False,
        title=dict(text=f'<b>Task 3: Test Set Evaluation & Error Taxonomy (N = {len(y_true)} Displays)</b><br><sup>Exact-Match Accuracy = {exact_acc:.2f}% | CER = {cer*100:.2f}% | Character Accuracy = {(1-cer)*100:.2f}%</sup>'),
    )
    fig5.update_xaxes(title_text='Ground-Truth Sequence Length', row=1, col=1)
    fig5.update_yaxes(title_text='Exact-Match Accuracy (%)', range=[0, 112], row=1, col=1)
    fig5.update_xaxes(title_text='Outcome Category', row=1, col=2)
    fig5.update_yaxes(title_text='Number of Test Images', range=[0, max(exact_cnt, 10) * 1.15], row=1, col=2)
    save_fig(fig5, 'task3_evaluation_summary')

    # 6. Task 3 Sample Qualitative Predictions Grid (8 correct + up to 4 errors)
    ok_idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == p][:8]
    err_idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p][:4]
    chosen = ok_idx + err_idx
    if len(chosen) < 12:
        chosen += [i for i in range(len(y_true)) if i not in chosen][: 12 - len(chosen)]

    titles = []
    for i in chosen:
        if y_true[i] == y_pred[i]:
            titles.append(f'<span style="color:#2a9d8f"><b>[OK]</b></span> True: "{y_true[i]}"<br>Pred: "{y_pred[i]}"')
        else:
            titles.append(f'<span style="color:#e76f51"><b>[ERR]</b></span> True: "{y_true[i]}"<br>Pred: "{y_pred[i]}"')

    fig6 = make_subplots(rows=3, cols=4, subplot_titles=titles, vertical_spacing=0.18, horizontal_spacing=0.05)
    for pos, idx in enumerate(chosen):
        r, c = pos // 4 + 1, pos % 4 + 1
        img = np.asarray(Image.open(os.path.join(TASK3_IMG_DIR, filenames[idx])).convert('RGB').resize((256, 64)))
        fig6.add_trace(go.Image(z=img), row=r, col=c)
        fig6.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=r, col=c)
        fig6.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=r, col=c)

    fig6.update_annotations(font_size=11)
    fig6.update_layout(
        template='plotly_white', width=1280, height=700, margin=dict(t=110),
        title=dict(text='<b>Task 3: Qualitative Inference on Unseen 7-Segment Test Displays</b><br><sup>Representative Exact Matches and Remaining Edge-Case Errors</sup>'),
    )
    save_fig(fig6, 'task3_sample_predictions')


def _draw_pipeline_diagram(filename, title, subtitle, badges, top_row, bot_row, edges, summary_text):
    """Draw compact, crisp 2-row architecture diagram with cohesive color palettes."""
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
    fig, ax = plt.subplots(figsize=(12.2, 5.4), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis('off')

    # Header & top-right badges
    ax.text(1.0, 41.2, title, fontsize=14.5, fontweight='bold', color='#0f172a', va='center')
    ax.text(1.0, 38.4, subtitle, fontsize=10.3, color='#475569', va='center')

    bx = 99.0
    for b_text, b_fg, b_bg, b_ec in reversed(badges):
        bw = max(14.0, len(b_text) * 0.66 + 2.8)
        bx -= bw
        ax.add_patch(patches.FancyBboxPatch(
            (bx, 38.2), bw, 3.8, boxstyle='round,pad=0.2,rounding_size=0.5',
            facecolor=b_bg, edgecolor=b_ec, linewidth=1.4,
        ))
        ax.text(bx + bw / 2, 40.1, b_text, fontsize=10.2, fontweight='bold', color=b_fg, ha='center', va='center')
        bx -= 1.2

    def draw_box(x, y, w, h, st):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.6',
            facecolor=st['bg'], edgecolor=st['ec'], linewidth=1.8,
        ))
        cx = x + w / 2
        ax.text(cx, y + h - 2.2, st['title'], fontsize=11.5, fontweight='bold', color=st['fg'], ha='center', va='center')
        ax.text(cx, y + h - 4.7, st['line1'], fontsize=10.2, fontweight='semibold', color=st['fg'], ha='center', va='center')
        ax.text(cx, y + h - 7.1, st['line2'], fontsize=9.8, color=st['fg'], alpha=0.88, ha='center', va='center')

    # Compact 2-row grid: 3 boxes top (y=24.0, h=9.4), 3 boxes bottom (y=7.2, h=9.4)
    box_w, box_h = 27.0, 9.4
    xs = [1.5, 36.5, 71.5]
    positions = {}
    for idx, st in enumerate(top_row):
        draw_box(xs[idx], 24.0, box_w, box_h, st)
        positions[f'T{idx}'] = (xs[idx], 24.0, box_w, box_h)
    for idx, st in enumerate(bot_row):
        draw_box(xs[idx], 7.2, box_w, box_h, st)
        positions[f'B{idx}'] = (xs[idx], 7.2, box_w, box_h)

    # Arrows with clean connector labels
    for src, dst, label in edges:
        sx, sy, sw, sh = positions[src]
        dx, dy, dw, dh = positions[dst]
        if sy == dy and sx < dx:  # Left -> Right
            p1, p2 = (sx + sw + 0.3, sy + sh / 2), (dx - 0.3, dy + dh / 2)
            lx, ly = (p1[0] + p2[0]) / 2, p1[1] + 1.2
        elif sy == dy and sx > dx:  # Right -> Left
            p1, p2 = (sx - 0.3, sy + sh / 2), (dx + dw + 0.3, dy + dh / 2)
            lx, ly = (p1[0] + p2[0]) / 2, p1[1] + 1.2
        elif sx == dx and sy > dy:  # Top -> Bottom vertical
            p1, p2 = (sx + sw / 2, sy - 0.3), (dx + dw / 2, dy + dh + 0.3)
            lx, ly = p1[0] + 0.8, (p1[1] + p2[1]) / 2
        else:  # Diagonal
            p1, p2 = (sx + sw / 2, sy - 0.3), (dx + dw / 2, dy + dh + 0.3)
            lx, ly = (p1[0] + p2[0]) / 2 + 1.0, (p1[1] + p2[1]) / 2
        ax.annotate('', xy=p2, xytext=p1, arrowprops=dict(arrowstyle='-|>', color='#475569', lw=1.7, mutation_scale=14))
        if label:
            ax.text(
                lx, ly, label, fontsize=9.2, fontweight='semibold', color='#334155',
                ha='center' if sy == dy else 'left', va='center',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.9),
            )

    # Compact bottom summary bar
    ax.add_patch(patches.FancyBboxPatch(
        (1.0, 1.0), 98.0, 3.6, boxstyle='round,pad=0.15,rounding_size=0.5',
        facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.2,
    ))
    ax.text(2.4, 2.8, summary_text, fontsize=9.8, color='#334155', va='center')

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout(pad=0.25)
    plt.savefig(os.path.join(FIG_DIR, f'{filename}.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(FIG_DIR, f'{filename}.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {filename}.png and {filename}.svg')


def plot_diagrams():
    # Cohesive palettes: Slate (Input), Blue (Encoder/LSTM), Purple (Attention/Fusion), Green (Output)
    c_in = {'fg': '#1e293b', 'bg': '#f1f5f9', 'ec': '#475569'}
    c_enc = {'fg': '#1e40af', 'bg': '#dbeafe', 'ec': '#2563eb'}
    c_att = {'fg': '#6b21a8', 'bg': '#f3e8ff', 'ec': '#7e22ce'}
    c_out = {'fg': '#166534', 'bg': '#dcfce7', 'ec': '#16a34a'}

    # Diagram 1: Task 2 Seq2Seq LSTM + Attention
    _draw_pipeline_diagram(
        'task2_architecture',
        'Task 2: Encoder–Decoder LSTM + Attention Architecture',
        'Hourly Water Demand Forecasting (48h Lookback → 5h Forecast) • 121,025 Trainable Parameters',
        [('Params: 121,025', '#1e293b', '#f1f5f9', '#94a3b8'), ('Test R² = NSE: 0.9283', '#166534', '#dcfce7', '#16a34a')],
        [
            {'title': 'Input Window (48 × 9)', 'line1': '48 Past Hours × 9 Features', 'line2': 'Weather + Cyclical Sin/Cos', **c_in},
            {'title': 'LSTM Encoder (48 × 96)', 'line1': 'LSTM(96, seq=True, state=True)', 'line2': 'Exports enc_out + [h48, c48]', **c_enc},
            {'title': 'LSTM Decoder (5 × 96)', 'line1': 'RepeatVector(5) → LSTM(96)', 'line2': 'Initialized with [h48, c48]', **c_enc},
        ],
        [
            {'title': 'Output: 5 Horizons (t+1..t+5)', 'line1': 'Inverse Z-Score → m³/h', 'line2': 'MAE: 217.53 | RMSE: 291.41', **c_out},
            {'title': 'TimeDistributed Head (5 × 32)', 'line1': 'Dense(32, ReLU) → Dropout(0.1)', 'line2': 'Dense(1) → Reshape(5)', **c_att},
            {'title': 'Luong Attention (5 × 192)', 'line1': 'Query: Decoder · Key/Val: Encoder', 'line2': 'Concat [dec_out, context]', **c_att},
        ],
        [
            ('T0', 'T1', 'z-score'),
            ('T1', 'T2', '[h48, c48]'),
            ('T2', 'B2', 'dec_out'),
            ('T1', 'B2', 'enc_out (48×96)'),
            ('B2', 'B1', '5×192'),
            ('B1', 'B0', '5 scaled'),
        ],
        'Training Protocol: Chronological Split (70/15/15) • Train-Only StandardScaler • Adam (1e-3) + ReduceLROnPlateau • MSE Loss • Best Val Epoch: 10/17',
    )

    # Diagram 2: Task 3 CNN Encoder + LSTM Decoder + Multi-Head Attention
    _draw_pipeline_diagram(
        'task3_architecture',
        'Task 3: CNN–LSTM + Multi-Head Attention Architecture',
        '7SEG_OCR Sequence Recognition (8×64 Visual Resolution) • 886,735 Total Parameters',
        [('Params: 886,735', '#1e293b', '#f1f5f9', '#94a3b8'), ('Exact-Match: 91.32% | Char: 98.11%', '#166534', '#dcfce7', '#16a34a')],
        [
            {'title': 'Input Image (64 × 256 × 3)', 'line1': 'RGB 7-Segment LED Display', 'line2': 'Pixel Values Scaled [0, 1]', **c_in},
            {'title': 'CNN Backbone (8 × 64 × 128)', 'line1': '4× [Conv2D(3×3) + BatchNorm]', 'line2': 'Asym Pool (8×H, 4×W) Keeps Dots', **c_enc},
            {'title': 'Seq + Pos Encoder (64 × 128)', 'line1': 'Permute → Dense(128) → Conv1D(3)', 'line2': '+ Positional Embedding (0..63)', **c_enc},
        ],
        [
            {'title': 'Output Tokens (7 × 15)', 'line1': 'TimeDist Dense(64 → 15 Softmax)', 'line2': 'Greedy Decode → Stop at <EOS>', **c_out},
            {'title': 'Multi-Head Attention (7 × 256)', 'line1': 'MHA(4 Heads × d=32, Q=Dec)', 'line2': 'Concat [dec_out, ctx] + Drop(0.2)', **c_att},
            {'title': 'LSTM Decoder (7 × 128)', 'line1': 'MaxPool1D(8) → [h0, c0]', 'line2': 'TokEmb + PosEmb → LSTM(128)', **c_att},
        ],
        [
            ('T0', 'T1', 'RGB'),
            ('T1', 'T2', '64 slices'),
            ('T2', 'B2', 'init [h0, c0]'),
            ('T2', 'B1', 'K/V (64×128)'),
            ('B2', 'B1', 'dec_out (Q)'),
            ('B1', 'B0', 'logits'),
        ],
        'Training Protocol: Length-Stratified Split (80/10/10) • Teacher Forcing + <PAD> Sample-Weight Masking • CER: 1.89% • Digit/Sign Errors: 0/334',
    )


if __name__ == '__main__':
    plot_task2()
    plot_task3()
    plot_diagrams()

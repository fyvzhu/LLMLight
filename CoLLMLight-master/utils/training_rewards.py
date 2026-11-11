"""Pure response validation and rewards, shared by synthesis and PPO."""
import json
import re

SIGNALS = ('ETWT', 'NTST', 'ELWL', 'NLSL')


def parse_action(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            data, _ = decoder.raw_decode(text[match.start():])
            data = {str(k).lower(): v for k, v in data.items()}
            if data.get('signal') not in SIGNALS:
                continue
            used = data.get('used_atr', 0)
            if isinstance(used, bool) or not isinstance(used, int) or used not in (0, 1):
                continue
            return {'signal': data['signal'], 'reason': data.get('reason', ''), 'used_atr': used}
        except (ValueError, AttributeError):
            continue
    raise ValueError('No valid signal/used_atr JSON in response')


def calculate_reactive_action_reward(text, best_signal):
    try:
        action = parse_action(text)
        return (1.0 if action['signal'] == best_signal else -1.0), action['used_atr']
    except ValueError:
        return -1.0, 0


def calculate_advance_reasoning_reward(text, ra_reward, used_atr):
    # Preserve original topic-count proxy, not token length.
    length = min(len(re.findall(r'^##\s', text, re.MULTILINE)), 8)
    return ra_reward * (0.5 * (1 - length / 8) + 0.5 * used_atr)

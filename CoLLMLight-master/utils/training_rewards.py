"""One RA extraction and validation rule for SFT, PPO and CityFlow."""
import json
import re

SIGNALS = ('ETWT', 'NTST', 'ELWL', 'NLSL')


def parse_action(text):
    """Use the published JSON extraction order and validate the action fields.

    Missing used_atr retains the upstream default 0; otherwise it must be the
    JSON integer 0 or 1. Booleans, floats and numeric strings are not integers
    under this protocol. Missing reason remains optional.
    """
    if not isinstance(text, str):
        raise ValueError('RA response must be text')
    match = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL)
    if not match:
        match = re.search(r'({.*})', text.strip(), re.DOTALL)
    if not match:
        raise ValueError('No RA JSON object in response')
    try:
        data = json.loads(match.group(1))
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid RA JSON') from exc
    if not isinstance(data, dict):
        raise ValueError('RA JSON must be an object')
    data = {str(key).lower(): value for key, value in data.items()}
    if data.get('signal') not in SIGNALS:
        raise ValueError('Invalid or missing RA signal')
    used = data.get('used_atr', 0)
    if type(used) is not int or used not in (0, 1):
        raise ValueError('used_atr must be the integer 0 or 1')
    return {'signal': data['signal'], 'reason': data.get('reason', ''), 'used_atr': used}


def extract_json_reactive_action_results(response):
    """Official non-raising interface, using exactly the same validation rule."""
    try:
        return parse_action(response)
    except ValueError:
        return {'signal': None, 'reason': None, 'used_atr': 0}

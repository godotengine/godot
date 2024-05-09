import { expect } from 'chai';;
import Module from './_build/JsMaterialXCore.js';

describe('Environ', () =>
{
    let mx;
    before(async () =>
    {
        mx = await Module();
    });

    it('Environment variables', () =>
    {
        expect(mx.getEnviron(mx.MATERIALX_SEARCH_PATH_ENV_VAR)).to.equal('');
        mx.setEnviron(mx.MATERIALX_SEARCH_PATH_ENV_VAR, 'test');
        expect(mx.getEnviron(mx.MATERIALX_SEARCH_PATH_ENV_VAR)).to.equal('test');
        mx.removeEnviron(mx.MATERIALX_SEARCH_PATH_ENV_VAR);
        expect(mx.getEnviron(mx.MATERIALX_SEARCH_PATH_ENV_VAR)).to.equal('');
    });
});

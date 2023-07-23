#ifndef _SAMPLER_HPP_
#define _SAMPLER_HPP_

/*
def sample_necleus(self, logits):
        probs = F.softmax(logits.float(), dim=-1)
        sorted_ids = torch.argsort(probs)
        sorted_probs = probs[sorted_ids]
        sorted_probs = torch.flip(sorted_probs, dims=(0,))
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(
            cumulative_probs > self.top_p)])
        probs[probs < cutoff] = 0
        if self.top_k < len(probs) and self.top_k > 0:
            probs[sorted_ids[:-self.top_k]] = 0
        if self.temp != 1.0:
            probs = probs ** (1.0 / self.temp)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)

def sample_typical(self, logits):
        probs = F.softmax(logits.float(), dim=-1)
        logits = -torch.log(probs)
        entropy = torch.nansum(logits * probs, dim=-1, keepdim=True)
        logits = torch.abs(logits - entropy)

        sorted_ids = torch.argsort(logits)
        sorted_logits = logits[sorted_ids]
        sorted_probs = probs[sorted_ids]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = np.sum(cumulative_probs < self.tau)
        probs[logits > sorted_logits[cutoff]] = 0
        if self.temp != 1.0:
            probs = probs ** (1.0 / self.temp)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)
*/

#include <stdint.h>
#include <math.h>

#include "simd.hpp"

#include "kernel/softmax.cuh"
#include "thrust.hpp"

namespace sampler
{
    inline uint32_t typical(std::vector<float> logits, float tau, float temp)
    {
        // probs = F.softmax(logits.float(), dim=-1)
        std::vector<float> probs = logits;
        //simd::softmax(probs);
        cuda::softmax_(&probs[0], &probs[0], 1, probs.size());

        // logits = -torch.log(probs)
        logits = probs;
        simd::log(logits);
        simd::mul(logits, -1.f, logits);

        // entropy = torch.nansum(logits * probs, dim=-1, keepdim=True)
        std::vector<float> tmp(logits.size());
        simd::mul(logits, probs, tmp);
        auto entropy = simd::sum_all(tmp);

        // logits = torch.abs(logits - entropy)
        simd::sub(logits, entropy, logits);
        simd::abs(logits);
        
        // sorted_ids = torch.argsort(logits)
        std::vector<uint32_t> sorted_id(logits.size());
        //simd::argsort(sorted_id, logits);
        thrust::argsort_host(&sorted_id[0], &logits[0], logits.size());

        // sorted_logits = logits[sorted_ids]
        // sorted_probs = probs[sorted_ids]
        std::vector<float> sorted_prob(logits.size());
        std::vector<float> sorted_logits(logits.size());
        for (uint32_t i = 0; i < logits.size(); i++)
        {
            sorted_prob[i] = probs[sorted_id[i]];
            sorted_logits[i] = logits[sorted_id[i]];
        }
        //sorted_prob = probs;
        //sorted_logits = logits;

        // cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        simd::cumsum(tmp, sorted_prob);
        //thrust::cumsum_host(&tmp[0], &sorted_prob[0], sorted_prob.size());

        // cutoff = np.sum(cumulative_probs < self.tau)
        simd::cmp_lt(tmp, tau, tmp);
        auto cut_off = simd::sum_all(tmp);

        // probs[logits > sorted_logits[cutoff]] = 0
        auto cut_off_val = sorted_logits[cut_off];
        simd::cmp_le(logits, cut_off_val, tmp);
        simd::mul(probs, tmp, probs);

        if (temp != 1.f)
            // probs = probs ** (1.0 / self.temp)
            simd::pow(probs, 1.f / temp, probs);

        // out = torch.multinomial(probs, num_samples=1)[0]
        return simd::multinomial_one(probs);
    }
}


#endif
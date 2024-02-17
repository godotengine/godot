# Overcoming a V-Sekai limit with ewb-ik

## Metadata

- Status: Proposed
- Deciders: V-Sekai
- Tags: V-Sekai

## The Backdrop

V-Sekai is currently facing a limitation in its inverse kinematics system solver for multi-chain skeletons and with constraints. This system is crucial for creating realistic movements, especially in complex scenarios such as dance.

## The Challenge

The current system has multiple plausible solutions, which can lead to unpredictable results. Moreover, it requires manual configuration of cost regions in kusudama constraints, which can be time-consuming and error-prone. Additionally, integrating these changes into a production game presents another layer of complexity and potential risk.

## The Strategy

We propose to overcome this limitation by implementing an energy minimization approach in the ewb-ik version of the system. This approach will automatically define 0 cost regions along with hard boundary (high cost regions), resulting in the optimal solution with minimal effort from the user.

Additionally, we propose to add a requirement for zero configuration to make the system more user-friendly and efficient.

## The Upside

This strategy will not only improve the accuracy of the system but also enhance its usability. Users will no longer need to manually configure the system, saving them time and reducing the risk of errors.

## The Downside

Implementing this strategy may require significant changes to the existing system and could potentially introduce new bugs. Moreover, merging these changes into a production game could present additional challenges. However, we believe that the benefits outweigh the potential risks.

## The Road Not Taken

An alternative approach could be to improve the current system without implementing energy minimization. However, this would not address the issue of multiple plausible solutions and would still require manual configuration.

## The Infrequent Use Case

In rare cases, users may prefer to manually configure the system to achieve specific results. We should consider providing an option for advanced users to override the automatic configuration if needed.

## In Core and Done by Us?

Yes, this proposal involves core changes to the system and will be implemented by our team.

## Further Reading

- [V-Sekai] - AI assisted this article.
